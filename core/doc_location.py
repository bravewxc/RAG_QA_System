# -*- coding: utf-8 -*-
"""措施26：文档位置精确定位（页码 + 行号标注）

解决什么问题：
    检索命中的 chunk 缺少"在原文哪里"的位置信息，回答无法精确溯源到"第几页第几行"。
    页级信息 PyPDFLoader 天然提供（metadata.page，0 起始，随切分自动继承），
    但行级信息在文本切分后丢失——chunk 只是文本片段，不知道自己对应原文第几行。

实现思路（零文本侵入：chunk 内容与不标注时逐字节一致，只新增 metadata 字段）：
    1. 行索引：把切分前的基准文本（PDF 为逐页文本）按换行符切行，
       记录每行的字符区间 [start, end)，得到"字符偏移 → 行号"映射
    2. 父chunk定位：父chunk是基准文本的连续子串（splitter 不修改内容，仅可能去首尾空白），
       按切分顺序用 find 顺序查找（起点推进防回退误匹配），得到字符区间 → 相交行号范围
    3. 子chunk定位：子chunk是父chunk文本的子串，先在父chunk内定位相对偏移，
       加上父chunk绝对偏移得到绝对区间 → 行号；并钳制在父chunk行范围内，限制误差传播

"行"的口径（生产上必须说清）：
    PDF：指 pypdf 提取文本按换行符切分的页内行（近似版面行）。年报这类数字版 PDF
    版面简单，行号与阅读版面的行基本一一对应；扫描件或复杂多栏版面存在偏差。
    TXT/MD/DOCX：指加载器提取文本的行号——TXT 与原文件行一致；MD/DOCX 经过
    unstructured 重组（如 Markdown 表格管道符合并为空格分隔的一行），行号是
    "提取文本行"而非原始文件行，同页/同段落内容可能落在同一行号上。
    升级路径：换 pdfplumber.extract_text_lines（带 top/bottom 坐标）即可做到
    "版面精确行 + 前端坐标高亮"，本模块的行索引结构已为其预留。

已知边界（诚实声明）：
    - 任何一步定位失败都回退为父chunk的行范围（页内 ±几行的误差，绝不丢页）
    - 高度重复的短行（表格合计行等）理论上可能顺序查找错位，钳制机制保证不越出父chunk范围
"""
from loguru import logger


def build_line_index(text: str) -> list:
    """构建行索引：[(start, end, line_no)]。

    行号从 1 开始（与编辑器/Word 习惯一致），end 不含换行符本身。
    空行也会占一个行号（与编辑器语义一致）。
    """
    index = []
    pos = 0
    for line_no, line in enumerate(text.split("\n"), start=1):
        index.append((pos, pos + len(line), line_no))
        pos += len(line) + 1  # +1 为换行符本身
    return index


def locate(text: str, chunk: str, search_from: int = 0):
    """在 text 中从 search_from 起查找 chunk 的字符区间。

    返回 (start, end)（end 为开区间边界）；找不到返回 None。
    顺序查找 + 起点推进：chunk 按切分顺序出现，下一个 chunk 的起点必然
    不早于上一个的起点，从上次位置起找可避免在重复文本处回退误匹配。
    """
    if not chunk:
        return None
    start = text.find(chunk, search_from)
    if start == -1:
        return None
    return start, start + len(chunk)


def line_range_for(line_index: list, start: int, end: int):
    """返回与字符区间 [start, end) 相交的行号闭区间 (line_start, line_end)。"""
    hits = [no for (s, e, no) in line_index if s < end and e > start]
    if not hits:
        return None
    return min(hits), max(hits)


def annotate_pdf_parents(parent_docs: list, page_docs: list) -> list:
    """PDF：给父chunk标注页内行号，返回与 parent_docs 等长的定位上下文列表。

    上下文（供子chunk映射用）：{'char_start', 'line_index', 'line_range'} 或 None。
    行号是页内行号（每页从第1行重新数起），配合 metadata.page 一起引用才有意义。
    """
    # 每页只建一次行索引（PDF切分前后 page 不会重复出现，但防御性去重）
    page_texts = {}
    page_line_indices = {}
    for d in page_docs:
        pg = d.metadata.get("page")
        if isinstance(pg, int) and pg not in page_texts:
            page_texts[pg] = d.page_content
            page_line_indices[pg] = build_line_index(d.page_content)

    contexts = []
    last_start = {}  # 每页独立的顺序查找起点
    for doc in parent_docs:
        pg = doc.metadata.get("page")
        text = page_texts.get(pg)
        if text is None:
            contexts.append(None)
            continue
        r = locate(text, doc.page_content, last_start.get(pg, 0))
        if r is None:
            r = locate(text, doc.page_content, 0)  # 兜底：从头再找一次
        if r is None:
            # 理论不可达（父chunk必然来自该页文本），防御性跳过
            logger.debug(f"措施26: 父chunk在页文本中未定位到, page={pg}")
            contexts.append(None)
            continue
        last_start[pg] = r[0]
        lr = line_range_for(page_line_indices[pg], *r)
        if lr:
            doc.metadata["line_start"], doc.metadata["line_end"] = lr
        contexts.append({
            "char_start": r[0],
            "line_index": page_line_indices[pg],
            "line_range": lr,
        })
    return contexts


def annotate_flat_parents(parent_docs: list, full_text: str) -> list:
    """无页码文档（TXT/MD/DOCX 等单文档加载器）：全文级行号标注。

    与 PDF 版的区别：没有页概念，行号是全文档行号；上下文结构与 PDF 版一致，
    子chunk映射逻辑完全复用。
    """
    line_index = build_line_index(full_text)
    contexts = []
    last_start = 0
    for doc in parent_docs:
        r = locate(full_text, doc.page_content, last_start)
        if r is None:
            r = locate(full_text, doc.page_content, 0)
        if r is None:
            contexts.append(None)
            continue
        last_start = r[0]
        lr = line_range_for(line_index, *r)
        if lr:
            doc.metadata["line_start"], doc.metadata["line_end"] = lr
        contexts.append({
            "char_start": r[0],
            "line_index": line_index,
            "line_range": lr,
        })
    return contexts


def locate_child_line_range(parent_text: str, parent_loc: dict, child_text: str,
                            search_from: int = 0):
    """子chunk行号定位：在父chunk内定位相对偏移 → 换算绝对偏移 → 查行号 → 钳制。

    返回 (line_range, next_search_from)；定位失败返回 (None, search_from)，
    调用方此时让子chunk继承父chunk的行范围（已在 metadata 里）。
    """
    if parent_loc is None or not child_text.strip():
        return None, search_from
    rel = locate(parent_text, child_text, search_from)
    if rel is None:
        return None, search_from
    abs_start = parent_loc["char_start"] + rel[0]
    abs_end = parent_loc["char_start"] + rel[1]
    lr = line_range_for(parent_loc["line_index"], abs_start, abs_end)
    # 钳制：无论相对定位是否错位，行号绝不越出父chunk的行范围
    pr = parent_loc.get("line_range")
    if lr and pr:
        lr = (max(lr[0], pr[0]), min(lr[1], pr[1]))
    return lr, rel[0]
