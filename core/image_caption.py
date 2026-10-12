# -*- coding: utf-8 -*-
"""措施25：PDF 图片/图表多模态理解入库

解决什么问题：
    PyPDFLoader 只提取 PDF 文本层，年报中的图表（营收趋势图、业务占比饼图等）
    以及图片格式的表格内容会静默丢失——"看图才能回答"的问题无法作答，且无任何报错提示。

生产级 RAG 的图片处理思路（四步流水线，本项目按此实现）：
    1. 检测：定位每页中的图片区域（两类信号：光栅图片 bbox / 矢量绘图密集区）
    2. 渲染：把该区域按 150ppi 渲染为 PNG（矢量图必须走"渲染"而非"抠图"，否则拿不到）
    3. 理解：调用多模态模型（qwen-vl-plus）生成结构化文字描述（图表类型/标题/关键数值/结论）
    4. 入库：描述文本作为独立 chunk 进入向量库 + BM25，metadata 记录页码与原图路径，
       后续提问即可命中图片内容，回答可溯源到"某文件第N页的图表"

成本控制（生产环境必须考虑，多模态调用按次计费）：
    - 尺寸过滤：跳过图标/logo/页眉装饰类小图（短边 < IMAGE_MIN_SIDE_PT）
    - 数量上限：每页最多 IMAGE_MAX_PER_PAGE 张、每个 PDF 最多 IMAGE_MAX_PER_PDF 张
    - 文本密度过滤：矢量密集但文字多的页是"表格页"（已有措施21 JSON 旁路），跳过避免重复
    - 内容哈希去重：同一图片（如每页重复的 logo）只处理第一次出现
    - 描述缓存：caption 以 JSON 落盘在 PNG 旁，重建索引时直接复用，不重复调用多模态模型；
      与 SQLRecordManager 增量索引配合——未变化的图片 chunk 在 index() 阶段被 num_skipped 跳过

失败降级：
    - 单张图片失败只跳过该图片；整个模块异常由调用方 try/except 兜底，绝不影响原文本入库
"""
import base64
import hashlib
import io
import json
import os
import re

import pdfplumber
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from loguru import logger

from infra.config import settings

# 图片/说明缓存落盘目录（chroma/image_store/，与向量库同级，便于整体备份迁移）
IMAGE_STORE_DIR = settings.IMAGE_STORE_DIR

# 渲染分辨率（ppi）：150 对 A4 页面约 1240x1754 像素，图表数值清晰且 PNG 通常 < 500KB
_RENDER_RESOLUTION = 150
# 单张图片字节上限：超过说明渲染区域异常大，跳过（DashScope 多模态请求体有大小限制）
_MAX_PNG_BYTES = 6 * 1024 * 1024

# 多模态模型的图片说明提示词（财报场景定制：读数值、防编造）
_CAPTION_PROMPT = """你是财报图表分析助手。请用中文描述这张财报文档页面中的图片/图表，按以下格式输出：
【图表类型】柱状图/折线图/饼图/堆叠图/表格截图/流程图/照片/其他
【标题】图中标题文字（没有则写"无"）
【关键数据】图中可读出的关键数值，尽量给出具体数字与单位（如"2024年营业收入 7,771亿元"），逐条列出，最多8条；没有可读数值则写"图中无数值"
【一句话结论】这张图传达的核心信息
要求：只描述图中可见内容，看不清或读不出的不要编造。"""


def _candidate_boxes(page):
    """提取一页中的候选图片区域 bbox 列表（pdfplumber 坐标系: x0, top, x1, bottom，单位为 PDF 点）。

    两类检测信号：
    1. 光栅图片：page.images 给出内嵌位图的 bbox（年报中导出为图片的图表、照片）。
       过滤条件：短边 >= IMAGE_MIN_SIDE_PT 且面积 >= 页面 0.5%（排除 logo/图标/装饰线）。
    2. 矢量图表：PDF 里大量图表是矢量绘制的（线条/矩形/曲线），page.images 完全看不到，
       只能靠"矢量元素数量密集"来发现。同时要求页面文字少（<120 词）——
       矢量密集 + 文字密集的页通常是财务报表表格（边框线多），
       表格数据已有措施21结构化JSON旁路，无需重复处理。
    """
    boxes = []
    page_w, page_h = page.width, page.height

    # 信号1：光栅图片
    for im in page.images:
        w = im["x1"] - im["x0"]
        h = im["bottom"] - im["top"]
        if min(w, h) < settings.IMAGE_MIN_SIDE_PT:
            continue
        if w * h < page_w * page_h * 0.005:
            continue
        boxes.append((
            max(0.0, im["x0"]), max(0.0, im["top"]),
            min(page_w, im["x1"]), min(page_h, im["bottom"]),
        ))

    # 信号2：矢量图表密集页（无光栅图时才检测，避免重复）
    if not boxes:
        vec_objs = page.rects + page.lines + page.curves
        n_vec = len(vec_objs)
        if n_vec >= settings.IMAGE_VECTOR_OBJ_THRESHOLD:
            n_words = len(page.extract_words())
            if n_words < 120:
                # 取所有矢量元素的并集 bbox 作为图表区域
                xs0 = [o["x0"] for o in vec_objs]
                tops = [o["top"] for o in vec_objs]
                xs1 = [o["x1"] for o in vec_objs]
                bottoms = [o["bottom"] for o in vec_objs]
                union = (
                    max(0.0, min(xs0)), max(0.0, min(tops)),
                    min(page_w, max(xs1)), min(page_h, max(bottoms)),
                )
                # 并集几乎铺满整页时直接整页渲染（图表+图注文字一起理解）
                if (union[2] - union[0]) * (union[3] - union[1]) > page_w * page_h * 0.9:
                    boxes.append((0.0, 0.0, page_w, page_h))
                else:
                    boxes.append(union)

    return boxes


def _render_region(page, bbox) -> bytes:
    """把页面指定 bbox 区域渲染为 PNG 字节（150ppi）。矢量图表必须走渲染而非位图抠取。"""
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        raise ValueError(f"无效bbox: {bbox}")
    page_image = page.crop(bbox).to_image(resolution=_RENDER_RESOLUTION)
    buf = io.BytesIO()
    page_image.original.save(buf, format="PNG")
    return buf.getvalue()


def _caption_from_cache_or_vlm(png_path: str, png_bytes: bytes, chat) -> tuple:
    """生成图片说明：优先读缓存 JSON（重建索引零成本），未命中才调用多模态模型。

    缓存 key 是渲染 PNG 的内容哈希（文件名即哈希），因此同一图片跨页/跨文件重复出现
    或文档重复上传重建时，都不会产生重复的多模态调用。
    返回 (caption, from_cache)。
    """
    cache_path = f"{png_path}.json"
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                caption = json.load(f).get("caption", "")
            if caption:
                return caption, True
        except Exception as e:
            logger.warning(f"图片说明缓存读取失败，将重新生成: {cache_path}: {e}")

    b64 = base64.b64encode(png_bytes).decode("utf-8")
    message = HumanMessage(content=[
        {"type": "text", "text": _CAPTION_PROMPT},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ])
    resp = chat.invoke([message])
    caption = str(getattr(resp, "content", "") or "").strip()
    if not caption:
        raise ValueError("多模态模型返回空描述")

    # 写缓存（失败不影响主流程，只是下次重建会重新调用）
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"caption": caption, "model": settings.IMAGE_CAPTION_MODEL},
                      f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"图片说明缓存写入失败: {cache_path}: {e}")
    return caption, False


def _is_low_value_photo(caption: str) -> bool:
    """判断是否为低价值图片：厂区/产品照片且无可读数值。

    这类图片的说明不包含任何业务数据，入库只会成为噪声 chunk 稀释检索精度
    （图表/表格类说明即使数值不全也有定位价值，予以保留）。
    """
    return "照片" in caption and "图中无数值" in caption


def build_image_documents(pdf_path: str) -> list:
    """对单个 PDF 执行 检测→渲染→理解→组装 的完整流水线，返回图片说明 chunk 列表。

    返回的 Document 设计说明：
    - page_content: 【图片说明】+ 页码 + 多模态模型描述（检索与生成可直接使用）
    - metadata.source: 与文本 chunk 完全一致的文件路径（保证 index() 的 source_id_key 溯源一致）
    - metadata.page: 0 起始页码，与 PyPDFLoader 的 page 语义一致（与文本 chunk 可对齐）
    - metadata.parent_content/parent_id: 自映射——SmallToBigRetriever 召回子chunk后
      靠这两个字段映射回父chunk，图片说明没有"父chunk"，自映射避免被检索器丢弃
    - metadata.image_path: 原图 PNG 落盘路径，供前端后续展示原图
    """
    # 延迟导入：避免与 infra.models 的初始化顺序耦合
    from infra.models import get_lc_model_client

    chat = get_lc_model_client(
        model=settings.IMAGE_CAPTION_MODEL, max_tokens=1024, temperature=0.1
    )
    os.makedirs(IMAGE_STORE_DIR, exist_ok=True)
    fname = os.path.basename(pdf_path)

    # 从文件名提取报告期（与文本 chunk 的 report_period 语义保持一致）
    m = re.search(r"(20\d{2})年报", pdf_path)
    report_period = f"{m.group(1)}年报" if m else ""

    docs = []
    seen_hashes = set()   # 内容哈希去重：同一张图（如每页 logo）只入库第一次出现
    total = 0

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            if total >= settings.IMAGE_MAX_PER_PDF:
                break
            kept_on_page = 0
            for bbox in _candidate_boxes(page):
                if (kept_on_page >= settings.IMAGE_MAX_PER_PAGE
                        or total >= settings.IMAGE_MAX_PER_PDF):
                    break
                try:
                    png_bytes = _render_region(page, bbox)
                except Exception as e:
                    logger.warning(f"措施25: 区域渲染失败跳过 [{fname} p{page_idx}]: {e}")
                    continue
                if len(png_bytes) > _MAX_PNG_BYTES:
                    logger.warning(f"措施25: 渲染结果过大跳过 [{fname} p{page_idx}]: {len(png_bytes)}B")
                    continue

                content_hash = hashlib.md5(png_bytes).hexdigest()
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                # PNG 落盘：内容哈希命名 → 天然去重 + caption 缓存可长期复用
                png_path = os.path.join(IMAGE_STORE_DIR, f"{content_hash[:12]}.png")
                if not os.path.exists(png_path):
                    with open(png_path, "wb") as f:
                        f.write(png_bytes)

                try:
                    caption, from_cache = _caption_from_cache_or_vlm(png_path, png_bytes, chat)
                except Exception as e:
                    logger.warning(f"措施25: 图片理解失败跳过 [{fname} p{page_idx}]: {e}")
                    continue

                # 计数在"理解完成"时自增：上限的本质是控制多模态调用次数（成本），
                # 即使后续被低价值过滤跳过，调用成本也已经发生
                kept_on_page += 1
                total += 1
                logger.info(f"措施25: 图片说明{'命中缓存' if from_cache else '已生成(调用多模态模型)'} "
                            f"[{fname} p{page_idx}] {len(caption)}字 → {png_path}")

                # 低价值图片过滤：照片类且无可读数值（厂区/产品照）不入库
                if _is_low_value_photo(caption):
                    logger.debug(f"措施25: 低价值图片跳过入库 [{fname} p{page_idx}]")
                    continue

                content = f"【图片说明】（{fname} 第{page_idx + 1}页）{caption}"
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": pdf_path,
                        "page": page_idx,
                        "source_type": "image",
                        "image_path": png_path,
                        "report_period": report_period,
                        "statement_type": "图表",
                        "parent_content": content,
                        "parent_id": hashlib.md5(content.encode("utf-8")).hexdigest(),
                    },
                ))

    return docs
