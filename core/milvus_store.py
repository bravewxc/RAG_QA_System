# -*- coding: utf-8 -*-
"""Milvus 向量存储适配。

对上层仍暴露 LangChain VectorStore/Retriever 接口，业务层无需感知底层由
Chroma 切换为 Milvus。底层通过 langchain-milvus 使用官方 pymilvus.MilvusClient。

默认连接 Milvus Standalone/分布式服务；也支持将 MILVUS_URI 配置为本地 .db
文件（需要运行环境额外支持 Milvus Lite）。
"""
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import Milvus
from loguru import logger

from infra.config import settings
from infra.retry_utils import with_milvus_retry


def get_physical_collection_name(logical_name: str) -> str:
    """将业务 collection 名转换为合法的 Milvus collection 名。

    原业务 collection 名是知识库目录的 MD5，可能以数字开头；Milvus 要求名称
    以字母或下划线开头，因此统一增加 kb_ 前缀。外部仍只使用原逻辑名称。
    """
    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", logical_name)
    safe_name = safe_name[:220]
    return f"kb_{safe_name}"


def _is_local_file_uri(uri: str) -> bool:
    return uri.endswith(".db") and not uri.startswith(("http://", "https://"))


def _build_connection_args() -> Dict[str, Any]:
    """构造 pymilvus.MilvusClient 参数，不回传任何认证信息到日志。"""
    uri = settings.MILVUS_URI
    args: Dict[str, Any] = {"uri": uri, "timeout": settings.MILVUS_TIMEOUT}

    if _is_local_file_uri(uri):
        absolute_path = os.path.abspath(uri)
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
        args["uri"] = absolute_path
        return args

    args["db_name"] = settings.MILVUS_DB_NAME
    args["secure"] = settings.MILVUS_SECURE
    if settings.MILVUS_TOKEN:
        args["token"] = settings.MILVUS_TOKEN
    else:
        if settings.MILVUS_USER:
            args["user"] = settings.MILVUS_USER
        if settings.MILVUS_PASSWORD:
            args["password"] = settings.MILVUS_PASSWORD
    return args


class ResilientMilvus(Milvus):
    """为写入、检索和删除增加统一重试及操作日志的 Milvus VectorStore。"""

    @with_milvus_retry
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        timeout: Optional[float] = None,
        batch_size: int = 1000,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        text_list = list(texts)
        logger.debug(
            f"Milvus 写入开始: collection={self.collection_name}, "
            f"texts={len(text_list)}, batch_size={batch_size}"
        )

        if ids and self.col is not None:
            # 已有 collection 时使用 upsert，确保超时重试不会重复插入同一主键。
            metadata_list = metadatas or [{} for _ in text_list]
            documents = [
                Document(page_content=text, metadata=metadata_list[i])
                for i, text in enumerate(text_list)
            ]
            self.upsert(
                ids=ids,
                documents=documents,
                batch_size=batch_size,
                timeout=timeout,
                **kwargs,
            )
            result = list(ids)
        else:
            # 新 collection 首次写入由父类完成建集合、建索引和加载。
            result = super().add_texts(
                text_list,
                metadatas=metadatas,
                timeout=timeout,
                batch_size=batch_size,
                ids=ids,
                **kwargs,
            )
        logger.debug(
            f"Milvus 写入完成: collection={self.collection_name}, inserted={len(result)}"
        )
        return result

    @with_milvus_retry
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        param: Optional[dict | list[dict]] = None,
        expr: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        logger.debug(
            f"Milvus 检索开始: collection={self.collection_name}, k={k}, expr={expr}"
        )
        result = super().similarity_search_with_score(
            query,
            k=k,
            param=param,
            expr=expr,
            timeout=timeout,
            **kwargs,
        )
        logger.debug(
            f"Milvus 检索完成: collection={self.collection_name}, hits={len(result)}"
        )
        return result

    @with_milvus_retry
    def delete(
        self,
        ids: Optional[List[str]] = None,
        expr: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        logger.debug(
            f"Milvus 删除开始: collection={self.collection_name}, "
            f"ids={len(ids) if ids else 0}, expr={expr}"
        )
        return super().delete(ids=ids, expr=expr, **kwargs)


@with_milvus_retry
def create_milvus_store(
    collection_name: str,
    embedding_function: Embeddings,
) -> ResilientMilvus:
    """创建 Milvus VectorStore，保持原 create_indexes 的调用方式。

    - HNSW + COSINE 对应原 Chroma 的语义检索配置；
    - auto_id=False，使增量索引可以继续使用内容+metadata 的 SHA256 作为稳定主键；
    - enable_dynamic_field=True，保留 source/page/report_period/parent_content 等元数据；
    - Strong 一致性保证写入和增量删除后能够立即查询到结果。
    """
    physical_name = get_physical_collection_name(collection_name)
    connection_args = _build_connection_args()
    logger.info(
        f"初始化 Milvus: collection={physical_name}, "
        f"uri={connection_args['uri']}, db={connection_args.get('db_name', 'milvus-lite')}"
    )

    store = ResilientMilvus(
        embedding_function=embedding_function,
        collection_name=physical_name,
        collection_description=f"Knowledge base vectors for {collection_name}",
        connection_args=connection_args,
        consistency_level=settings.MILVUS_CONSISTENCY_LEVEL,
        index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {
                "M": settings.MILVUS_HNSW_M,
                "efConstruction": settings.MILVUS_HNSW_EF_CONSTRUCTION,
            },
        },
        search_params={
            "metric_type": "COSINE",
            "params": {"ef": settings.MILVUS_HNSW_EF},
        },
        auto_id=False,
        primary_field="pk",
        text_field="text",
        vector_field="vector",
        enable_dynamic_field=True,
    )
    # langchain-milvus 0.2.2 在新 collection 首次写入时会将 self.timeout 与
    # kwargs.timeout 重复传给 MilvusClient.insert。连接级超时仍由 connection_args
    # 生效，这里清空 VectorStore 操作级默认值以避免重复关键字。
    store.timeout = None
    return store
