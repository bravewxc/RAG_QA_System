# -*- coding: utf-8 -*-
"""统一重试机制：对 LLM/Embedding/Rerank 三类调用封装指数退避重试。

策略：
- 针对可重试错误（429 限流、5xx 服务端错误、TimeoutError）进行指数退避重试
- 重试 3 次，退避 1s/2s/4s
- 其他错误（4xx 客户端错误、参数错误等）不重试，直接抛出
- 超时熔断：连续重试耗尽后抛出最后一次异常

接入方式：
- LLM：在 ChatOpenAI 外层包一层带重试的 invoke/stream
- Embedding：在 DashScopeEmbeddings 的 embed_query/embed_documents 外层加重试
- Rerank：在 SafeDashScopeRerank.rerank 外层加重试
"""
import functools
from typing import Callable, TypeVar

import grpc
import openai
from loguru import logger
from pymilvus.exceptions import MilvusException
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
    retry_if_exception_type,
    before_sleep_log,
)

T = TypeVar("T")

# 可重试的异常类型：429 限流、5xx 服务端错误、超时
RETRYABLE_EXCEPTIONS = (
    openai.RateLimitError,        # 429
    openai.APIStatusError,        # 5xx 会走这里（APIStatusError 是 5xx 的基类之一）
    openai.APITimeoutError,       # 请求超时
    openai.APIConnectionError,    # 连接错误
    TimeoutError,                 # 通用超时
    ConnectionError,              # 通用连接错误
)


def _is_retryable_api_error(exc: Exception) -> bool:
    """判断 openai API 异常是否可重试（429 或 5xx）。

    openai 的 APIStatusError 带 status_code 属性，429/5xx 可重试，
    其他 4xx（400/401/403/404 等）不可重试。
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        return False
    return status_code == 429 or status_code >= 500


def _retryable_predicate(exc: Exception) -> bool:
    """tenacity 重试谓词：仅对可重试异常重试。

    对 openai.APIStatusError，需进一步判断 status_code（只有 429/5xx 才重试）。
    """
    if isinstance(exc, RETRYABLE_EXCEPTIONS):
        if isinstance(exc, openai.APIStatusError) and not _is_retryable_api_error(exc):
            return False
        return True
    return False


# 统一重试装饰器：3 次，指数退避 1s/2s/4s
def with_retry(func: Callable[..., T]) -> Callable[..., T]:
    """为 LLM/Embedding/Rerank 调用添加统一重试。

    - 429/5xx/超时/连接错误：重试 3 次，退避 1s/2s/4s
    - 其他错误：不重试，直接抛出
    - 重试前打印 WARNING 日志，便于排查
    """
    @retry(
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),  # 1s, 2s, 4s
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except openai.APIStatusError as e:
            # 对 4xx 非 429 错误不重试，直接抛出
            if not _is_retryable_api_error(e):
                logger.error(f"不可重试的API错误 status={e.status_code}: {e}")
                raise
            logger.warning(f"可重试的API错误 status={e.status_code}, 即将重试: {e}")
            raise
        except RETRYABLE_EXCEPTIONS as e:
            logger.warning(f"可重试异常, 即将重试: {type(e).__name__}: {e}")
            raise
    return wrapper


def is_retryable_milvus_error(exc: Exception) -> bool:
    """判断 Milvus/gRPC 异常是否值得重试。

    仅重试限流、服务端暂不可用、连接中断和超时；collection/schema/表达式/
    主键等参数或数据错误直接抛出，避免无效重试掩盖真实问题。
    """
    if isinstance(exc, grpc.RpcError):
        return exc.code() in {
            grpc.StatusCode.DEADLINE_EXCEEDED,
            grpc.StatusCode.UNAVAILABLE,
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            grpc.StatusCode.ABORTED,
            grpc.StatusCode.INTERNAL,
        }

    if not isinstance(exc, MilvusException):
        return False

    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in {429, 500, 502, 503, 504}:
        return True

    message = str(exc).lower()
    transient_keywords = (
        "timeout",
        "timed out",
        "deadline exceeded",
        "unavailable",
        "connection reset",
        "connection refused",
        "temporarily",
        "rate limit",
        "resource exhausted",
        "service unavailable",
    )
    return any(keyword in message for keyword in transient_keywords)


def with_milvus_retry(func: Callable[..., T]) -> Callable[..., T]:
    """为 Milvus 初始化、写入、检索和删除增加 3 次指数退避重试。"""

    @retry(
        retry=retry_if_exception(is_retryable_milvus_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        try:
            return func(*args, **kwargs)
        except (MilvusException, grpc.RpcError) as e:
            retryable = is_retryable_milvus_error(e)
            log = logger.warning if retryable else logger.error
            log(
                f"Milvus 调用失败: func={func.__name__}, retryable={retryable}, "
                f"error={type(e).__name__}: {e}"
            )
            raise

    return wrapper
