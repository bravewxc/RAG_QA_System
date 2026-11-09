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

    # Milvus 向量库（默认连接本机 Standalone，可通过 URI/认证信息切换云上实例）
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_DB_NAME: str = "default"
    MILVUS_TOKEN: str = ""       # 优先使用 token/API Key；为空时使用用户名密码
    MILVUS_USER: str = ""
    MILVUS_PASSWORD: str = ""
    MILVUS_SECURE: bool = False
    MILVUS_TIMEOUT: float = 30.0
    MILVUS_CONSISTENCY_LEVEL: str = "Strong"
    MILVUS_HNSW_M: int = 16
    MILVUS_HNSW_EF_CONSTRUCTION: int = 64
    MILVUS_HNSW_EF: int = 64

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

    # 图片/图表多模态处理（措施25）
    IMAGE_CAPTION_ENABLED: bool = True          # 总开关：False 时完全跳过图片处理（纯文本行为）
    IMAGE_CAPTION_MODEL: str = "qwen-vl-plus"   # 多模态理解模型（DashScope 兼容模式）
    IMAGE_STORE_DIR: str = "./chroma/image_store"  # 原图PNG + 说明缓存JSON 落盘目录
    IMAGE_MIN_SIDE_PT: float = 100.0            # 光栅图短边最小值(PDF点)，过滤logo/图标/装饰
    IMAGE_MAX_PER_PAGE: int = 4                 # 每页最多处理的图片数
    IMAGE_MAX_PER_PDF: int = 30                 # 每个PDF最多处理的图片数（控制多模态调用成本）
    IMAGE_VECTOR_OBJ_THRESHOLD: int = 25        # 矢量元素数阈值：超过视为图表页（矢量绘图检测）

    # 文档位置精确定位（措施26）：chunk metadata 注入页码+行号，支持"第N页第M-K行"级溯源
    LINE_LOCATE_ENABLED: bool = True


settings = Settings()
