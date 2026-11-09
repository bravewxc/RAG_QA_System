import hashlib
import os
import shutil
import threading
from typing import Optional, List

import jieba
from langchain_classic.indexes import SQLRecordManager
from langchain_classic.retrievers import ContextualCompressionRetriever, RePhraseQueryRetriever, EnsembleRetriever, MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import LLMChainFilter, CrossEncoderReranker
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.indexing import index
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from core.custom_loader import MyCustomLoader
from core.milvus_store import create_milvus_store
from infra.models import get_lc_model_client, get_ali_embeddings, get_ali_rerank
from loguru import logger


# 措施18：原始query并行双路检索器
# 并行执行「原始query → ensemble」与「重写query → ensemble」两路检索，
# 按 page_content 哈希去重合并后返回，供后续 rerank 统一排序。
class DualQueryRetriever(BaseRetriever):
    """双路检索器：原始query + RePhraseQuery重写query 并行检索，结果合并去重。

    与已回退的措施8（MultiQuery，LLM生成3-5个随机变体，噪声过多）不同：
    本检索器只做确定性的2路并行——原始query一路 + 重写query一路，
    噪声增量极小，且由后续 rerank 兜底过滤。
    """

    raw_retriever: BaseRetriever  # 原始query走的ensemble检索器
    rewrite_retriever: BaseRetriever  # 重写query走的RePhraseQuery检索器

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # 第一路：原始query直接走ensemble（保留精确数字与口径关键词）
        try:
            raw_docs = self.raw_retriever.invoke(query)
        except Exception as e:
            logger.warning(f"措施18: 原始query检索异常: {e}")
            raw_docs = []
        # 第二路：重写query走RePhraseQuery（语义扩展）
        try:
            rewrite_docs = self.rewrite_retriever.invoke(query)
        except Exception as e:
            logger.warning(f"措施18: 重写query检索异常: {e}")
            rewrite_docs = []

        # 按 page_content 哈希去重合并（保留首次出现的doc，原始query优先）
        seen = set()
        merged = []
        for d in raw_docs + rewrite_docs:
            key = hashlib.md5(d.page_content.encode("utf-8")).hexdigest()
            if key not in seen:
                seen.add(key)
                merged.append(d)
        logger.debug(f"措施18: 原始{len(raw_docs)}+重写{len(rewrite_docs)}→合并去重{len(merged)}")
        return merged


# 措施15：父子分层索引 Small-to-Big 检索器
# 检索子chunk后，通过 metadata.parent_content 映射回父chunk，按 parent_id 去重返回父chunk。
# 目的：小粒度子chunk让向量排序更准（CP），父chunk覆盖周边上下文（CR）。
class SmallToBigRetriever(BaseRetriever):
    """Small-to-Big 检索器：召回子chunk → 映射回父chunk → 去重返回。"""

    base_retriever: BaseRetriever  # 底层检索器（索引的是子chunk）

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        child_docs = self.base_retriever.invoke(query)
        # 子chunk → 父chunk映射，按 parent_id 去重
        seen_parent_ids = set()
        parent_docs = []
        for child in child_docs:
            parent_content = child.metadata.get('parent_content')
            parent_id = child.metadata.get('parent_id')
            if parent_content and parent_id and parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                # 构造父chunk Document（继承子chunk的metadata，但page_content替换为父chunk内容）
                parent_meta = {k: v for k, v in child.metadata.items()
                               if k not in ('parent_content', 'parent_id')}
                parent_docs.append(Document(page_content=parent_content, metadata=parent_meta))
        logger.debug(f"措施15: 子chunk {len(child_docs)} → 父chunk {len(parent_docs)}（去重后）")
        return parent_docs

# 财务领域专有名词词典，防止 jieba 将其拆散（如"归属于上市公司股东的净利润"不应被拆为"归属于/上市公司/股东/的/净利润"）
_FINANCIAL_TERMS = [
    "归属于上市公司股东的净利润", "归属于上市公司股东的扣除非经常性损益的净利润",
    "经营活动产生的现金流量净额", "加权平均净资产收益率", "基本每股收益",
    "非经常性损益", "营业收入", "营业成本", "营业利润", "利润总额",
    "货币资金", "应收账款", "应付账款", "存货", "固定资产", "在建工程",
    "无形资产", "商誉", "递延所得税资产", "递延所得税负债", "预计负债",
    "资产负债率", "流动比率", "速动比率", "毛利率", "净利率",
    "合并资产负债表", "合并利润表", "合并现金流量表", "公司资产负债表", "公司利润表",
    "关键审计事项", "审计报告", "审计意见", "注册会计师", "安永华明",
    "会计政策", "会计估计", "追溯调整", "未来适用法", "收入确认",
    "套期保值", "衍生品", "关联交易", "内部控制", "公司治理",
    "比亚迪", "刀片电池", "仰望", "腾势", "方程豹",
    "动力电池", "新能源汽车", "汽车分部", "手机部件",
]
for _term in _FINANCIAL_TERMS:
    jieba.add_word(_term)


def _tokenize_zh(text: str) -> str:
    """用 jieba 对中文文本分词，词间以空格分隔，供 BM25 检索使用。

    财报文本几乎不含空格，BM25 默认按空格分词会导致中文退化为整句匹配。
    jieba 分词后，专有名词（已加载词典）保持完整，通用词按语义切分。
    """
    return " ".join(jieba.cut(text))


def _jieba_preprocessing_func(text: str) -> List[str]:
    """BM25 的中文分词预处理函数，替代默认的按空格切分。

    BM25Retriever 内部对文档和查询都调用此函数做分词，
    返回的是分词后的 list（而非空格拼接的字符串）。
    """
    return list(jieba.cut(text))

# 设置知识库 向量模型 重排序模型的路径
KNOWLEDGE_DIR = './chroma/knowledge/'          # 知识库源文件目录（目录即知识库，保留原数据路径）
VECTOR_STORE_DIR = './milvus/vector_store'     # 可选 Milvus Lite 文件目录；远程 Milvus 模式不使用
embedding_model = get_ali_embeddings()

class MyKnowledge:
    """
    知识库管理模块
    """
    # 向量化模型
    __embeddings = embedding_model
    logger.info(f"当前嵌入模型: {__embeddings.model}")

    __retrievers = {}
    __llm = get_lc_model_client()

    # 每个知识库一把锁：保证同一知识库的建索引只发生一次（防懒加载与后台预热并发竞争）
    __kb_locks: dict = {}
    __kb_locks_guard = threading.Lock()

    def _get_kb_lock(self, collection_name):
        """获取指定知识库的构建锁（按 collection_name 粒度）"""
        with self.__kb_locks_guard:
            lock = self.__kb_locks.get(collection_name)
            if lock is None:
                lock = threading.Lock()
                self.__kb_locks[collection_name] = lock
            return lock

    def upload_knowledge(self, temp_file, collection):
        """
        处理原始文档的上传：将文件上传到指定知识库目录（collection 即目录名），
        并清除该库的检索器缓存，下次提问/预热时自动重建索引
        """
        if not collection:
            # 未选择知识库时，自动归入第一个知识库目录（默认库），保证上传不失败
            dirs = sorted(d for d in os.listdir(KNOWLEDGE_DIR)
                          if os.path.isdir(os.path.join(KNOWLEDGE_DIR, d)))
            if not dirs:
                logger.warning("上传失败：尚无任何知识库目录")
                import gradio as gr
                gr.Warning("上传失败：chroma/knowledge/ 下还没有知识库目录，请先创建（如 合同库/制度库/报表库）")
                return None, gr.update(choices=self.load_knowledge())
            collection = dirs[0]
            logger.warning(f"未选择知识库，默认上传至 [{collection}]")
            import gradio as gr
            gr.Info(f"未选择知识库，文件默认上传至 [{collection}]，可在知识库下拉框中切换")

        file_name = os.path.basename(temp_file)
        dest_dir = os.path.join(KNOWLEDGE_DIR, collection)
        os.makedirs(dest_dir, exist_ok=True)
        file_path = os.path.join(dest_dir, file_name)
        # 如果文件不存在就copy
        if not os.path.exists(file_path):
            shutil.copy(temp_file, file_path)
            logger.info(f"已上传文件 [{file_name}] 到知识库 [{collection}]")

        # 该库内容已变化，移除缓存的检索器
        self.__retrievers.pop(get_md5(collection), None)

        # 上传后立即向量化入库：后台线程立即重建该库索引（不等首次提问）。
        # 注意不能通过 get_retrievers 触发重建：若此刻懒加载/预热线程刚好在重建该库
        # 并已把检索器写回缓存，get_retrievers 会命中缓存短路（不重新扫描目录），
        # 导致新上传的文件不被索引（表现为 num_skipped 不变、num_added=0）。
        # 因此这里加锁强制重建，并与懒加载/预热共用同一把锁，避免并发覆盖缓存。
        def _index_now():
            try:
                logger.info(f"上传后开始为知识库 [{collection}] 建立索引...")
                collection_md5 = get_md5(collection)
                lock = self._get_kb_lock(collection_md5)
                with lock:
                    dir_path = self._find_knowledge_dir(collection)
                    if dir_path is None:
                        logger.warning(f"知识库目录不存在，无法建索引: {collection}")
                        return
                    built = create_indexes(collection_md5, dir_path, self.__embeddings)
                    if built is None:
                        logger.warning(f"知识库 [{collection}] 无可用文档，未建立检索器")
                        return
                    self.__retrievers[collection_md5] = built
                logger.info(f"知识库 [{collection}] 索引建立完成")
            except Exception as e:
                logger.error(f"知识库 [{collection}] 上传后建索引失败: {e}")
        threading.Thread(target=_index_now, name="kb-index-after-upload", daemon=True).start()

        import gradio as gr
        gr.Info(f"文件已上传至 [{collection}]，正在后台建立索引，完成后即可提问")
        return None, gr.update(choices=self.load_knowledge())

    def load_knowledge(self):
        # exist_ok=True目标目录已存在的情况下不会抛出异常。
        # 这意味着如果目录已经存在，os.makedirs不会做任何事情，也不会报错
        os.makedirs(os.path.dirname(KNOWLEDGE_DIR), exist_ok=True)

        # 知识库默认为空
        collections = [None]
        logger.info(f"当前知识库目录列表: {os.listdir(KNOWLEDGE_DIR)}")

        # 目录即知识库：仅列出 KNOWLEDGE_DIR 下的子目录名（如 合同库/制度库/报表库），
        # 目录内的所有文件合并为一个知识库；不在此处加载切分、不建索引（懒加载见 get_retrievers）
        for name in sorted(os.listdir(KNOWLEDGE_DIR)):
            if os.path.isdir(os.path.join(KNOWLEDGE_DIR, name)):
                collections.append(name)
        return collections

    def _find_knowledge_dir(self, dir_name):
        """在知识库根目录下查找子目录路径（目录即知识库，供懒加载建索引使用）"""
        path = os.path.join(KNOWLEDGE_DIR, dir_name)
        if os.path.isdir(path):
            return path
        return None

    def get_retrievers(self, collection):
        collection_name = get_md5(collection)
        if collection_name not in self.__retrievers:
            # 加锁 + 二次检查：懒加载与后台预热线程可能同时访问同一知识库，
            # 保证建索引只执行一次
            lock = self._get_kb_lock(collection_name)
            with lock:
                if collection_name not in self.__retrievers:
                    dir_path = self._find_knowledge_dir(collection)
                    if dir_path is None:
                        logger.warning(f"知识库目录不存在，无法建索引: {collection}")
                        return None
                    logger.info(f"懒加载知识库 [{collection}]，开始加载切分与建索引...")
                    built = create_indexes(collection_name, dir_path, self.__embeddings)
                    if built is None:
                        logger.warning(f"知识库 [{collection}] 无可用文档，未建立检索器")
                        return None
                    self.__retrievers[collection_name] = built
                    logger.info(f"知识库 [{collection}] 检索器构建完成")

        retriever = self.__retrievers[collection_name]
        logger.info(f"get_retrievers中: {retriever}")
        """
        ContextualCompressionRetriever:在上下文中压缩和优化检索结果
        结合了基础压缩器（base_compressor）和基础检索器（base_retriever），
        以减少不相关信息，返回更为精炼的检索结果。
        """
        # 措施12：移除 LLMChainFilter（评测提速 + 避免 LLM 误删有用 chunk）
        # 问题：LLMChainFilter 对每个候选 chunk 逐条调 LLM 判断相关性，137 条评测产生
        #       ~1370-2740 次额外 LLM 调用，是检索阶段最大耗时瓶颈；且财务数字相关性判断
        #       对 LLM 较难，可能误删有用 chunk（如"营业成本"与"毛利率"相关但被判不相关）。
        # 改动：原结构 RePhraseQuery → LLMChainFilter(逐chunk调LLM) → rerank
        #       新结构 RePhraseQuery → rerank（仅靠 rerank score 排序 + top_n=5 截断）
        # 措施17曾尝试 rerank 分数阈值0.3截断，但 CR -0.028、F -0.060（误删弱相关但必要的chunk），
        # 效果为负已回退为不截断（min_score_threshold=None）
        # 措施22：rerank 后 soft boost —— 对 statement_type 匹配 query 关键词的 chunk 加分（×1.1），
        # 不排除任何 chunk，仅提升匹配项排序，保底召回不受影响（区别于措施10的 hard filter）
        _boost_map = {
            '审计意见': ['审计报告'], '审计机构': ['审计报告'], '签字注册会计师': ['审计报告'],
            '关键审计事项': ['审计报告'], 'KAM': ['审计报告'],
            '会计政策': ['会计政策'], '收入确认': ['会计政策', '审计报告'],
            '存货': ['会计政策'], '折旧': ['会计政策'], '固定资产': ['会计政策'],
            '管理层讨论': ['管理层讨论'], '经营情况': ['管理层讨论'],
        }
        # 措施24：rerank top_n 从5降至4，去掉排序最低的弱相关/无关chunk，提升CP
        rerank_retriever = get_ali_rerank(top_n=4, boost_keywords_map=_boost_map)

        # 措施15+18 联合实施：父子分层索引 Small-to-Big + 原始query并行双路检索
        # 措施15：SmallToBigRetriever 包装 ensemble（索引的是子chunk，检索后映射回父chunk去重返回）
        #   - 子chunk（400字符）语义聚焦，rerank打分更准 → 提升CP
        #   - 父chunk（1500字符）上下文完整 → 保障CR和生成质量
        # 措施18：DualQueryRetriever 包装 SmallToBigRetriever（原始query + 重写query 双路并行）
        #   - 原始query保留精确数字与口径关键词，修复重写丢词导致的检索失败
        #   - 重写query做语义扩展，覆盖更多相关chunk
        #   - 两路结果按page_content哈希去重合并，统一送rerank精排
        # 联合动机：措施15单独实施时F退化（父chunk更大让模型倾向回答而非拒答），
        #   措施18单独实施时F退化（双路引入额外上下文让模型读错数字）。
        #   两者联合期望：子chunk精准检索 + 双路召回提升CP/CR，
        #   同时子chunk的精准映射减少噪声父chunk数量，缓解F退化。
        small_to_big = SmallToBigRetriever(base_retriever=retriever)
        dual_retriever = DualQueryRetriever(
            raw_retriever=small_to_big,
            rewrite_retriever=RePhraseQueryRetriever.from_llm(small_to_big, self.__llm)
        )
        final_retriever = ContextualCompressionRetriever(
            base_compressor=rerank_retriever,
            base_retriever=dual_retriever
        )
        logger.debug(f"最终检索器为: {final_retriever}")
        return final_retriever

    def get_raw_retriever(self, collection):
        """
        返回原始 ensemble retriever（bypass RePhraseQuery + rerank）。
        供措施9「含数字query精确补充检索」使用：当问题含具体财务数字时，
        RePhraseQuery 可能改写丢失数字，用原始 query 直接走 BM25 精确匹配补充。
        """
        collection_name = get_md5(collection)
        if collection_name not in self.__retrievers:
            self.get_retrievers(collection)
        return self.__retrievers.get(collection_name)

    def warmup_indexes(self):
        """
        后台线程预热：启动后静默为所有知识库构建检索器（复用懒加载 get_retrievers）。
        不阻塞主线程，用户提问时若目标库尚未预热完成，则照常懒加载（并发安全见 get_retrievers 锁）
        """
        def _warmup():
            try:
                dirs = [d for d in os.listdir(KNOWLEDGE_DIR)
                        if os.path.isdir(os.path.join(KNOWLEDGE_DIR, d))]
                logger.info(f"后台预热开始: 共 {len(dirs)} 个知识库")
                for d in dirs:
                    if self.get_retrievers(d) is None:
                        logger.warning(f"预热跳过（无文档或建索引失败）: {d}")
                logger.info("后台预热完成")
            except Exception as e:
                logger.error(f"后台预热异常: {e}")

        threading.Thread(target=_warmup, name="kb-warmup", daemon=True).start()

# 创建索引
def create_indexes(collection_name: str, dir_path: str,
                   embedding_function: Optional[Embeddings] = None):
    # Milvus VectorStore：HNSW + COSINE，保留 LangChain VectorStore/Retriever 调用接口。
    # 物理 collection 名由适配层添加 kb_ 前缀，外部仍使用知识库目录 MD5。
    if embedding_function is None:
        embedding_function = get_ali_embeddings()
    db = create_milvus_store(collection_name, embedding_function)

    # 记录管理器沿用 SQLite；命名空间切换为 milvus，避免误用旧 Chroma 指纹记录。
    record_manager = SQLRecordManager(
        f"milvus/{collection_name}", db_url="sqlite:///db/record_manager_cache.db"
    )
    """
    在文档被索引到 Milvus 之前，SQLRecordManager 管理文档的索引记录。
    它确保文档的索引状态被正确记录，避免重复索引或遗漏。
    """
    logger.info(f"record_manager: {record_manager}")
    # 初始化数据库表结构
    """
    在开始加载和索引文档之前，调用create_schema() 方法，确保用于存储文档元数据和索引信息的数据库表已经存在。
    如果是第一次运行这个脚本，create_schema()方法会自动创建这些表。
    对于后续运行，如果表已经存在，则该方法不会重复创建，而是直接通过。
    """
    record_manager.create_schema()

    # 目录级知识库：加载目录下所有支持的文件，合并切分后统一索引（一个目录 = 一个知识库）
    logger.info(f"准备进行文件的加载切分.... 目录: {dir_path}")
    documents = []
    for file in sorted(os.listdir(dir_path)):
        file_path = os.path.join(dir_path, file)
        if not os.path.isfile(file_path):
            continue
        try:
            loader = MyCustomLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            logger.warning(f"跳过无法解析的文件 [{file}]: {e}")
    logger.info(f"文档切分数量: {len(documents)}....")
    logger.debug(f"documents: {documents}")

    if not documents:
        logger.warning(f"知识库目录 [{dir_path}] 无可用文档，跳过索引构建")
        return None

    # 将加载的文档索引到数据库中
    r = index(documents, record_manager, db, cleanup="full", source_id_key="source", key_encoder="sha256")
    logger.info(f"文档索引结果为: {r}")
    """
    num_added: 0 表示没有新文档被添加到数据库中。
    num_updated: 0 表示没有文档被更新。
    num_skipped: 8 表示有8个文档被跳过，没有被索引到数据库中。 提高索引效率
    num_deleted: 0 表示没有文档被删除。
    """
    '''混合检索，将稀疏检索器（如BM25）与密集检索器（如嵌入相似性）相结合。
    稀疏检索器擅长根据关键字查找相关文档，而密集检索器擅长根据语义相似性查找相关文档。
    BM25 使用 jieba 中文分词（含财务专有名词词典），替代默认的按空格切分。'''
    ensemble_retriever = EnsembleRetriever(
        # 密集检索 k=10（从3提升），BM25 k=10（从4提升），扩大召回候选数
        retrievers=[
            db.as_retriever(search_kwargs={"k": 10}),
            BM25Retriever.from_documents(documents, k=10, preprocess_func=_jieba_preprocessing_func)
        ]
        # 注：措施6曾尝试 weights=[0.4, 0.6]（密集0.4, BM25 0.6），但评测显示 CP -0.050、F -0.062，
        # 提高BM25权重牺牲了向量语义匹配质量，效果为负，已回退为默认 [0.5, 0.5]
    )
    # logger.info(f"ensemble_retriever: {ensemble_retriever}")
    return ensemble_retriever

def get_md5(input_string):
    # 创建一个 md5 哈希对象
    hash_md5 = hashlib.md5()
    # 需要确保输入字符串是字节串，因此如果它是字符串，则需要编码为字节串
    hash_md5.update(input_string.encode('utf-8'))
    # 获取十六进制的哈希值
    return hash_md5.hexdigest()
