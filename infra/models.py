import os
import threading

from langchain_community.document_compressors import DashScopeRerank
from langchain_openai import ChatOpenAI
import inspect
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from typing import Any, Dict, List, Optional, Sequence, Union

from infra.retry_utils import with_retry
from infra.config import settings

# DashScope 并发控制：用 Semaphore 限制同时调用的 LLM/Embedding/Rerank 数量，
# 避免触发 DashScope 的 RPM 限制（qwen 预览模型 QPS 限流严格）。
# 信号量在线程间共享，acquire/release 线程安全。
_dashscope_semaphore = threading.Semaphore(settings.DASHSCOPE_MAX_CONCURRENCY)


class _ConcurrencyLimitedChatOpenAI(ChatOpenAI):
    """ChatOpenAI 子类：在 invoke/stream 前后用 Semaphore 控制 DashScope 并发。

    与 max_retries（SDK内部重试）配合：
    - Semaphore 控制并发数，避免突发请求触发 RPM 限制
    - max_retries 处理偶发 429/5xx
    """

    def invoke(self, input, config=None, **kwargs):
        _dashscope_semaphore.acquire()
        try:
            return super().invoke(input, config=config, **kwargs)
        finally:
            _dashscope_semaphore.release()

    def stream(self, input, config=None, **kwargs):
        _dashscope_semaphore.acquire()
        try:
            return super().stream(input, config=config, **kwargs)
        finally:
            _dashscope_semaphore.release()

ALI_TONGYI_API_KEY_OS_VAR_NAME = "DASHSCOPE_API_KEY"
ALI_TONGYI_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _get_dashscope_api_key() -> Optional[str]:
    """API key 解析优先级：OS 环境变量 > 本地 .env（经 pydantic-settings 读取）。

    两者都没有时返回 None，与原先仅用 os.getenv 的行为一致。
    """
    return os.getenv(ALI_TONGYI_API_KEY_OS_VAR_NAME) or settings.DASHSCOPE_API_KEY or None
ALI_TONGYI_MAX_MODEL = "qwen3.8-flash"
ALI_TONGYI_DEEPSEEK_V4_FLASH = "deepseek-v4-flash-0731"
ALI_TONGYI_DEEPSEEK_V3 = "deepseek-v3"
ALI_TONGYI_REASONER_MODEL = "qvq-max-latest"
ALI_TONGYI_EMBEDDING_MODEL = "text-embedding-v4"
ALI_TONGYI_RERANK_MODEL = "gte-rerank-v2"


class SafeDashScopeRerank(DashScopeRerank):
    """自定义 DashScopeRerank，修复两个问题：
    1. 空字符串文档会导致 API 返回 400 错误: "document index:N should not be empty"
    2. API 返回错误时 output 为 None，原始代码直接访问 results.output.results 会抛 AttributeError

    措施17：增加 rerank 分数阈值自适应截断（动态 top_n）
    - 利用 gte-rerank-v2 已返回的 relevance_score（0-1）做动态截断
    - 保留 relevance_score >= 阈值 的 chunk，上限 top_n，下限保底1
    - 阈值通过 min_score_threshold 参数配置，默认 None（不截断，保持向后兼容）
    """

    # 措施17：rerank 分数阈值，None 表示不截断（向后兼容）
    min_score_threshold: Optional[float] = None

    # 措施22：rerank 后 soft boost 关键词映射
    # {query关键词: [匹配的 statement_type 列表]}，匹配的 chunk relevance_score *= 1.1
    zboost_keywords_map: Optional[Dict[str, List[str]]] = None

    @with_retry
    def rerank(
        self,
        documents: Sequence[Union[str, Document, dict]],
        query: str,
        *,
        top_n: Optional[int] = -1,
    ) -> List[Dict[str, Any]]:
        if len(documents) == 0:
            return []
        # query 为空或纯空白时直接返回，否则 DashScope rerank API 返回 400:
        # "Input should be a valid string: input.query"
        # 注意: query 可能是 list 类型（多 query 检索场景），需兼容处理
        if not query:
            return []
        if isinstance(query, str):
            if not query.strip():
                return []
        elif isinstance(query, (list, tuple)):
            query = " ".join(str(q) for q in query if q and str(q).strip())
            if not query:
                return []
        else:
            query = str(query)
            if not query.strip():
                return []
        docs = [
            doc.page_content if isinstance(doc, Document) else doc for doc in documents
        ]
        # 措施22：保存原始 Document 列表，供 soft boost 读取 metadata
        orig_documents = list(documents)
        # 过滤掉空字符串或纯空白文档，否则 DashScope rerank API 返回 400 错误
        indexed_docs = [(i, d) for i, d in enumerate(docs) if d and d.strip()]
        if len(indexed_docs) == 0:
            return []
        docs = [d for _, d in indexed_docs]
        orig_indices = [i for i, _ in indexed_docs]

        top_n = top_n if (top_n is None or top_n > 0) else self.top_n

        results = self.client.call(
            model=self.model,
            query=query,
            documents=docs,
            top_n=top_n,
            return_documents=False,
            api_key=self.dashscope_api_key,
        )

        if results.output is None:
            raise ValueError(
                f"DashScope rerank API returned empty output. "
                f"Status: {results.status_code}, Code: {results.code}, "
                f"Message: {results.message}, Request ID: {results.request_id}, "
                f"Model: {self.model}"
            )

        result_dicts = []
        for res in results.output.results:
            orig_index = orig_indices[res.index]
            result_dicts.append(
                {
                    "index": orig_index,
                    "relevance_score": res.relevance_score,
                }
            )

        # 措施22：rerank 后 soft boost（对 statement_type 匹配 query 关键词的 chunk 加分）
        # 不排除任何 chunk，仅对匹配项 relevance_score *= 1.1 并重新排序
        if self.boost_keywords_map and isinstance(query, str):
            for r in result_dicts:
                doc_idx = r['index']
                if doc_idx < len(orig_documents):
                    doc = orig_documents[doc_idx]
                    if isinstance(doc, Document):
                        stmt_type = doc.metadata.get('statement_type', '')
                        for kw, boost_types in self.boost_keywords_map.items():
                            if kw in query and stmt_type in boost_types:
                                r['relevance_score'] = r['relevance_score'] * 1.1
                                break
            result_dicts.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

        # 措施17：rerank 分数阈值自适应截断
        # 保留 relevance_score >= 阈值 的 chunk，上限 top_n，下限保底1
        if self.min_score_threshold is not None and len(result_dicts) > 0:
            filtered = [r for r in result_dicts
                        if r.get("relevance_score", 0) >= self.min_score_threshold]
            # 下限保底1，防止全部被滤掉时返回空（CP/CR/F 均会受影响）
            if len(filtered) == 0:
                filtered = result_dicts[:1]
            # 上限仍为 top_n（filtered 已是按分数降序，直接截断）
            result_dicts = filtered[:top_n]
        return result_dicts


def get_lc_model_client(api_key=None, base_url=ALI_TONGYI_URL
                        , model=ALI_TONGYI_MAX_MODEL, temperature=0.7, max_tokens=8000,verbose=False, debug=False,
                        max_retries=6, request_timeout=120):
    """
        通过LangChain获得指定平台和模型的客户端，设定的默认平台和模型为阿里百炼qwen-max-latest
        也可以通过传入api_key，base_url，model三个参数来覆盖默认值
        verbose，debug两个参数，分别控制是否输出调试信息，是否输出详细调试信息，默认不打印
        max_retries：请求失败（含 429 限流、5xx）时的自动重试次数，SDK 内部按指数退避重试，
                     默认 6 次，缓解预览模型 QPS 限流导致的偶发 429
        request_timeout：单次请求超时时间（秒），默认 120s
    """
    function_name = inspect.currentframe().f_code.co_name
    # 未显式传 key 时：OS 环境变量 > 本地 .env（settings），均为空则保持 None（报缺 key 的错）
    if api_key is None:
        api_key = _get_dashscope_api_key()
    if (verbose):
        print(f"{function_name}-平台：{base_url},模型：{model},温度：{temperature}")
    if (debug):
        _key_tail = api_key[-4:] if api_key else "N/A"
        print(f"{function_name}-平台：{base_url},模型：{model},温度：{temperature},key：****{_key_tail}")
    return _ConcurrencyLimitedChatOpenAI(api_key=api_key, base_url=base_url, model=model, temperature=temperature,
                      max_tokens=max_tokens, max_retries=max_retries, request_timeout=request_timeout)

def get_ali_embeddings():
    """通过LangChain获得一个阿里通义千问嵌入模型的实例"""
    return _RetryableDashScopeEmbeddings(
        model=ALI_TONGYI_EMBEDDING_MODEL, dashscope_api_key=_get_dashscope_api_key()
    )


class _RetryableDashScopeEmbeddings(DashScopeEmbeddings):
    """对 DashScopeEmbeddings 的 embed_query/embed_documents 加统一重试。

    DashScopeEmbeddings 底层调用 DashScope API，可能因 429/5xx/超时失败。
    通过 with_retry 装饰器实现 3 次指数退避重试（1s/2s/4s）。
    """

    @with_retry
    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(text)

    @with_retry
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents(texts)

def get_ali_rerank(top_n=10, min_score_threshold=None, boost_keywords_map=None):
    '''
    通过LangChain获得一个阿里重排序模型的实例
    :param top_n: 返回的 top N 文档数上限
    :param min_score_threshold: 措施17 rerank 分数阈值，None 表示不截断（向后兼容）
    :param boost_keywords_map: 措施22 soft boost 关键词映射，None 表示不 boost（向后兼容）
    :return: 阿里通义千问嵌入模型的实例
    '''
    rerank = SafeDashScopeRerank(
        model=ALI_TONGYI_RERANK_MODEL, dashscope_api_key=_get_dashscope_api_key(),
        top_n=top_n
    )
    # DashScopeRerank.validate_environment 会覆盖传入的 model 为默认的 gte-rerank（旧版），
    # 旧版模型返回 output=None 导致报错，此处强制恢复为 gte-rerank-v2
    rerank.model = ALI_TONGYI_RERANK_MODEL
    # 措施17：注入 rerank 分数阈值（默认 None 不截断，由调用方按需传入）
    rerank.min_score_threshold = min_score_threshold
    # 措施22：注入 soft boost 关键词映射（默认 None 不 boost）
    rerank.boost_keywords_map = boost_keywords_map
    return rerank


