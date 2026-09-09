# -*- coding: utf-8 -*-
"""全局配置管理：基于 pydantic-settings，从 .env 读取配置。

使用方式：
    from infra.config import settings
    model = settings.ANSWER_MODEL
"""
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # 阿里百炼 DashScope
    DASHSCOPE_API_KEY: str = ""

    # 模型配置
    ANSWER_MODEL: str = "deepseek-v4-flash-0731"
    EVAL_MODEL: str = "qwen3.8-flash"
    EMBEDDING_MODEL: str = "text-embedding-v4"
    RERANK_MODEL: str = "gte-rerank-v2"

    # 服务配置
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    GRADIO_HOST: str = "0.0.0.0"
    GRADIO_PORT: int = 7860

    # Chroma 向量库
    CHROMA_HOST: str = "chromadb"
    CHROMA_PORT: int = 8000
    CHROMA_PERSIST_DIR: str = "./chroma/vector_store"

    # Redis 缓存（预留，当前缓存用 SQLite）
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379

    # 重试配置
    RETRY_MAX_ATTEMPTS: int = 3
    RETRY_MIN_WAIT: float = 1.0
    RETRY_MAX_WAIT: float = 4.0

    # 限流配置
    DASHSCOPE_MAX_CONCURRENCY: int = 8
    API_RATE_LIMIT: str = "30/minute"

    # 可观测性（真实密钥放本地 .env，不入库）
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "doc-qa-system"
    LANGSMITH_TRACING: bool = False

    # 缓存配置
    CACHE_DB_PATH: str = "./db/cache/rag_cache.db"
    RETRIEVAL_CACHE_TTL: int = 3600
    SEMANTIC_CACHE_THRESHOLD: float = 0.92
    SEMANTIC_CACHE_TTL: int = 7200


settings = Settings()
