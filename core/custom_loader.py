from langchain_community.document_loaders import CSVLoader, TextLoader, UnstructuredWordDocumentLoader, PyPDFLoader, \
    UnstructuredMarkdownLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.file_utils.filetype import FileType, detect_filetype
from core.doc_location import annotate_pdf_parents, annotate_flat_parents, locate_child_line_range
from infra.config import settings
from loguru import logger
import hashlib
import re

# 措施16：三大主表行项目结构化重写
# 问题：PyPDFLoader 提取的表格文本列对齐丢失（如"货币资金  75,424,747  68,196,531"纯空格分隔），
#       模型在多列数字中读错列/行，导致数值精确性类 F 偏低（CR=1但F≤0.33）。
# 措施：识别三大主表区域，对表格行按多空格切分列，重写为结构化句式，
#       结构化文本置于原文之前，辅助模型读数（原文保留供忠实引用）。
# 三大主表章节标题（用于定位表格区域）
_FINANCIAL_STATEMENT_TITLES = [
    "合并资产负债表", "合并利润表", "合并现金流量表",
    "公司资产负债表", "公司利润表", "公司现金流量表",
]
# 财务数字正则：匹配含千分位逗号或小数点的数字（如 75,424,747 或 1,234.56 或 -123,456）
_FINANCIAL_NUMBER_PATTERN = re.compile(r'-?[\d,]+(?:\.\d+)?')

def _rewrite_financial_table_lines(text: str) -> str:
    """对三大主表区域的表格行做结构化重写。

    识别文本中的三大主表章节，对其中的表格行按多空格切分列，
    将"行项目  数字1  数字2"重写为"行项目：列1=数字1，列2=数字2（表名）"的结构化句式。
    结构化文本置于原文之前，原文保留。

    保守策略：
    - 只重写含至少2个数字列的行（避免误判标题行/空行）
    - 行项目名必须含中文字符（避免误判纯数字行）
    - 数字必须匹配财务数字格式（含千分位逗号或小数点，且长度>=3，避免误判页码等）
    """
    lines = text.split('\n')
    result_lines = []
    current_table = None  # 当前所在表名，None 表示不在主表区域
    rewritten_count = 0

    for line in lines:
        # 检测是否进入某个主表章节
        entered_table = None
        for title in _FINANCIAL_STATEMENT_TITLES:
            if title in line:
                entered_table = title
                break

        if entered_table:
            current_table = entered_table
            result_lines.append(line)
            continue

        # 简单的章节退出判断：遇到下一个主表标题或明显的非表格章节标题时退出
        # （保守起见，不主动退出，靠chunk切分自然限制范围）

        # 在主表区域内，尝试重写表格行
        if current_table and line.strip():
            # 按2个以上连续空格切分列
            parts = re.split(r' {2,}', line.strip())
            if len(parts) >= 3:  # 至少"行项目+2个数字列"
                row_name = parts[0].strip()
                num_parts = parts[1:]
                # 行项目名必须含中文字符
                if re.search(r'[\u4e00-\u9fff]', row_name):
                    # 提取数字部分，过滤非数字列
                    nums = []
                    for p in num_parts:
                        p = p.strip()
                        m = _FINANCIAL_NUMBER_PATTERN.search(p)
                        if m and len(m.group()) >= 3:  # 数字长度>=3，排除页码等
                            nums.append(m.group())
                    if len(nums) >= 2:  # 至少2个数字列才重写
                        # 构造结构化句式：行项目：本期=数字1，上期=数字2（表名）
                        col_labels = ["本期", "上期", "前年"]
                        parts_str = "，".join(
                            f"{col_labels[i] if i < len(col_labels) else f'第{i+1}列'}={nums[i]}"
                            for i in range(len(nums))
                        )
                        rewritten = f"{row_name}：{parts_str}（{current_table}）"
                        result_lines.append(rewritten)
                        rewritten_count += 1
                        continue
        result_lines.append(line)

    if rewritten_count > 0:
        logger.debug(f"措施16: 表格行结构化重写 {rewritten_count} 行（表: {current_table}）")
    return '\n'.join(result_lines)


class MyCustomLoader(BaseLoader):
    """
    文档加载和分割模块
    """
    # 支持加载的文件类型
    file_type = {
        FileType.CSV: (CSVLoader, {'autodetect_encoding': True}),
        FileType.TXT: (TextLoader, {'autodetect_encoding': True}),
        FileType.DOC: (UnstructuredWordDocumentLoader, {}),
        FileType.DOCX: (UnstructuredWordDocumentLoader, {}),
        FileType.PDF: (PyPDFLoader, {}),
        FileType.MD: (UnstructuredMarkdownLoader, {})
    }

    # 初始化方法，设置文档加载器和文本分割器
    def __init__(self, file_path: str):
        self.file_path = file_path  # 保存原始路径，供图片处理等措施25使用
        loader_class, params = self.file_type[detect_filetype(file_path)]
        logger.info(f"本文档[{file_path}]需使用文档加载器: {loader_class}")
        self.loader: BaseLoader = loader_class(file_path, **params)
        # 措施1+3：财报章节感知 + 表格保完整性切分
        # 问题：原 separators 仅 ["\n\n","\n"," ",""]，500字符盲切会在表格中间任意位置切断，
        #       导致同一张利润表的"营业收入"和"营业成本"分到不同chunk，跨行关联题证据不全。
        # 改动1（章节感知）：在 separators 前面增加财报章节标题作为优先分隔符，
        #   让切分优先发生在章节边界（如"合并资产负债表"、"关键审计事项"），保持每张报表/章节完整。
        # 改动2（表格保完整）：chunk_size 从 500 提升至 1000（容纳一张完整报表，三张主表通常不超过2000字符），
        #   chunk_overlap 从 200 提升至 300，减少报表行间断裂。
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=[
                # 财报章节标题优先切分（措施1）
                "\n合并资产负债表", "\n合并利润表", "\n合并现金流量表",
                "\n公司资产负债表", "\n公司利润表", "\n公司现金流量表",
                "\n关键审计事项", "\n审计报告", "\n管理层讨论与分析",
                "\n风险因素", "\n公司治理", "\n关联交易",
                # 通用分隔符
                "\n\n", "\n", " ", ""
            ],
            chunk_size=1500,        # 措施2：从1000提升至1500，容纳更完整报表
            chunk_overlap=300,      # 从200提升至300
            length_function=len,
        )

    def lazy_load(self):
        # 文档的切分加载
        # 注：load_and_split 内部即 load() + split_documents()，此处显式拆开，
        # 保留切分前的基准文档（PDF 为一页一个 Document），供措施26行号定位使用；
        # 切分结果与 load_and_split 完全一致（chunk 内容零变化）
        base_docs = self.loader.load()
        docs = self.text_splitter.split_documents(base_docs)
        # 措施10：从文件名提取报告期，注入 chunk metadata（供检索时按年份过滤）
        for d in docs:
            source = d.metadata.get('source', '')
            m = re.search(r'(20\d{2})年报', source)
            if m:
                d.metadata['report_period'] = f"{m.group(1)}年报"
        # 措施22：metadata 字段增强（statement_type/unit/scope），供检索后 soft boost 使用
        # 不改变 chunk 内容，仅注入 metadata，不影响向量分布
        for d in docs:
            text = d.page_content
            if any(kw in text for kw in ['合并资产负债表', '公司资产负债表']):
                d.metadata['statement_type'] = '资产负债表'
            elif any(kw in text for kw in ['合并利润表', '公司利润表']):
                d.metadata['statement_type'] = '利润表'
            elif any(kw in text for kw in ['合并现金流量表', '公司现金流量表']):
                d.metadata['statement_type'] = '现金流量表'
            elif any(kw in text for kw in ['关键审计事项', 'KAM']):
                d.metadata['statement_type'] = '审计报告'
            elif any(kw in text for kw in ['管理层讨论', '经营情况', '经营成果']):
                d.metadata['statement_type'] = '管理层讨论'
            elif '会计政策' in text:
                d.metadata['statement_type'] = '会计政策'
            elif '关联交易' in text:
                d.metadata['statement_type'] = '关联交易'
            elif any(kw in text for kw in ['公司治理', '股东大会', '董事会']):
                d.metadata['statement_type'] = '公司治理'
            else:
                d.metadata['statement_type'] = '其他'
            if '千元' in text:
                d.metadata['unit'] = '千元'
            elif '元）' in text or '元)' in text:
                d.metadata['unit'] = '元'
            else:
                d.metadata['unit'] = ''
            if '合并' in text:
                d.metadata['scope'] = '合并'
            elif any(kw in text for kw in ['母公司', '公司资产负债表', '公司利润表', '公司现金流量表']):
                d.metadata['scope'] = '母公司'
            else:
                d.metadata['scope'] = ''
        # 措施16曾尝试三大主表行项目结构化重写，但 CP -0.051、F -0.054
        # （结构化文本改变chunk内容稀释向量语义匹配，且与原文并存导致chunk变长让模型读数更难），
        # 效果为负已回退，保留 _rewrite_financial_table_lines 函数定义供后续优化参考

        # 措施26：位置精确定位（页码+行号）——先标注父chunk，子chunk在下方切分循环中映射
        # 定位失败整体降级为无行号入库，不影响文本 chunk 正常产出
        parent_locs = None
        if settings.LINE_LOCATE_ENABLED:
            try:
                if detect_filetype(self.file_path) == FileType.PDF:
                    parent_locs = annotate_pdf_parents(docs, base_docs)
                elif len(base_docs) == 1:
                    # TXT/MD/DOCX 等单文档加载器：全文级行号（无页码概念）
                    parent_locs = annotate_flat_parents(docs, base_docs[0].page_content)
                # CSV 等一行一个 Document 的加载器无版面行概念，跳过标注
            except Exception as e:
                logger.warning(f"措施26: 行号标注失败，本次按无行号入库 [{self.file_path}]: {e}")
                parent_locs = None

        # 措施15：父子分层索引 Small-to-Big
        # 子chunk（400字符）用于检索排序（语义更聚焦，rerank打分更准），
        # 父chunk（1500字符）用于生成（上下文更完整）。
        # 子chunk metadata 记录 parent_content 和 parent_id，
        # 检索时由 SmallToBigRetriever 映射回父chunk送入生成。
        # 任务1：子chunk切分separators改用中文切分符号，保留"\n\n","\n"，
        # 替换" ",""为中文句号/逗号/感叹号/问号/分号等，避免在中文词语中间硬切。
        child_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n", "\n",
                "。", "！", "？", "；", "，",
                " ", ""
            ],
            chunk_size=400,
            chunk_overlap=50,
            length_function=len,
        )
        child_docs = []
        for doc_idx, parent_doc in enumerate(docs):
            children = child_splitter.split_text(parent_doc.page_content)
            parent_id = hashlib.md5(parent_doc.page_content.encode('utf-8')).hexdigest()
            # 措施26：父chunk的定位上下文（None表示该父chunk未标注或标注被禁用）
            parent_loc = parent_locs[doc_idx] if parent_locs is not None else None
            search_from = 0  # 子chunk按切分顺序排列，顺序查找起点只进不退
            for child_text in children:
                child_meta = dict(parent_doc.metadata)
                child_meta['parent_content'] = parent_doc.page_content
                child_meta['parent_id'] = parent_id
                # 措施26：子chunk精确行号映射；失败则继承父chunk的行范围（已在metadata中）
                if parent_loc is not None:
                    lr, search_from = locate_child_line_range(
                        parent_doc.page_content, parent_loc, child_text, search_from)
                    if lr:
                        child_meta['line_start'], child_meta['line_end'] = lr
                child_docs.append(Document(page_content=child_text, metadata=child_meta))
        logger.info(f"措施15: 父chunk {len(docs)} → 子chunk {len(child_docs)}")

        # 措施25：图片/图表多模态理解入库
        # 仅对 PDF 处理：图表区域经"渲染→多模态模型生成文字说明"转为普通 chunk 入库。
        # 说明文本带内容哈希缓存（chroma/image_store/*.json），重建索引不会重复调用多模态模型；
        # 任何图片处理异常只影响图片 chunk，不影响上方文本 chunk 的正常返回。
        if settings.IMAGE_CAPTION_ENABLED and detect_filetype(self.file_path) == FileType.PDF:
            try:
                from core.image_caption import build_image_documents
                image_docs = build_image_documents(self.file_path)
                if image_docs:
                    child_docs.extend(image_docs)
                    logger.info(f"措施25: 注入图片说明 chunk {len(image_docs)} 条 [{self.file_path}]")
            except Exception as e:
                logger.warning(f"措施25: 图片处理失败，本文件按纯文本入库 [{self.file_path}]: {e}")

        return child_docs

    def load(self):
        # 加载
        return self.lazy_load()
