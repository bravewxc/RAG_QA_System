# -*- coding: utf-8 -*-
"""可观测性：LangSmith 全链路追踪 + 会话统计（Token/延迟/错误率）。

两部分：
1. LangSmith 追踪：通过环境变量 LANGSMITH_TRACING=true 启用，LangChain 自动上报
2. 会话统计：记录每次请求的 Token 消耗、检索延迟、生成延迟、错误率，写入 SQLite
"""
import os
import sqlite3
import threading
import time
from datetime import datetime
from typing import Optional

from loguru import logger

from infra.config import settings

# ============ LangSmith 全链路追踪 ============
def setup_langsmith_tracing():
    """启用 LangSmith 追踪（通过环境变量，LangChain 自动接入）。

    需设置环境变量：
    - LANGSMITH_TRACING=true
    - LANGSMITH_API_KEY=xxx
    - LANGSMITH_PROJECT=项目名
    """
    if not settings.LANGSMITH_TRACING:
        logger.info("LangSmith 追踪未启用（LANGSMITH_TRACING=false）")
        return
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    logger.info(f"LangSmith 追踪已启用, project={settings.LANGSMITH_PROJECT}")


# ============ 会话统计 ============
STATS_DB_PATH = "./db/cache/rag_stats.db"
_stats_conn = None
_stats_lock = threading.Lock()


def _get_stats_conn():
    global _stats_conn
    if _stats_conn is None:
        os.makedirs(os.path.dirname(STATS_DB_PATH), exist_ok=True)
        _stats_conn = sqlite3.connect(STATS_DB_PATH, timeout=30, check_same_thread=False)
        _stats_conn.execute("""
            CREATE TABLE IF NOT EXISTS request_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question TEXT,
                collection TEXT,
                model TEXT,
                retrieval_latency_ms INTEGER,
                generation_latency_ms INTEGER,
                total_latency_ms INTEGER,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                answer_length INTEGER,
                status TEXT,
                error_msg TEXT,
                created_at TEXT
            )
        """)
        _stats_conn.commit()
    return _stats_conn


class RequestStats:
    """单次请求统计上下文管理器"""

    def __init__(self, session_id: str, question: str, collection: str, model: str):
        self.session_id = session_id
        self.question = question
        self.collection = collection
        self.model = model
        self.start_time = time.time()
        self.retrieval_start: Optional[float] = None
        self.retrieval_end: Optional[float] = None
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.answer_length: int = 0
        self.status: str = "success"
        self.error_msg: str = ""

    def mark_retrieval_start(self):
        self.retrieval_start = time.time()

    def mark_retrieval_end(self):
        self.retrieval_end = time.time()

    def set_tokens(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens

    def mark_error(self, error_msg: str):
        self.status = "error"
        self.error_msg = error_msg[:500]

    def save(self):
        """保存统计到SQLite"""
        now = time.time()
        retrieval_ms = int((self.retrieval_end - self.retrieval_start) * 1000) if self.retrieval_start and self.retrieval_end else 0
        generation_ms = int((now - self.retrieval_end) * 1000) if self.retrieval_end else int((now - self.start_time) * 1000)
        total_ms = int((now - self.start_time) * 1000)

        with _stats_lock:
            conn = _get_stats_conn()
            conn.execute(
                """INSERT INTO request_stats
                (session_id, question, collection, model, retrieval_latency_ms, generation_latency_ms,
                 total_latency_ms, prompt_tokens, completion_tokens, total_tokens, answer_length,
                 status, error_msg, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (self.session_id, self.question[:200], self.collection, self.model,
                 retrieval_ms, generation_ms, total_ms,
                 self.prompt_tokens, self.completion_tokens,
                 self.prompt_tokens + self.completion_tokens,
                 self.answer_length, self.status, self.error_msg,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            conn.commit()
        logger.info(
            f"统计已记录: session={self.session_id}, total={total_ms}ms, "
            f"retrieval={retrieval_ms}ms, tokens={self.prompt_tokens + self.completion_tokens}, "
            f"status={self.status}"
        )


def get_stats_summary(hours: int = 1):
    """获取最近N小时的统计摘要"""
    with _stats_lock:
        conn = _get_stats_conn()
        rows = conn.execute(
            """SELECT
                COUNT(*) as total_requests,
                SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as error_count,
                AVG(total_latency_ms) as avg_latency_ms,
                AVG(retrieval_latency_ms) as avg_retrieval_ms,
                AVG(total_tokens) as avg_tokens,
                SUM(total_tokens) as total_tokens
            FROM request_stats
            WHERE created_at > datetime('now', ?)""",
            (f"-{hours} hours",),
        ).fetchone()
    if not rows or rows[0] == 0:
        return {"total_requests": 0, "error_rate": 0}
    total, success, error, avg_lat, avg_ret, avg_tok, sum_tok = rows
    return {
        "total_requests": total,
        "success_count": success or 0,
        "error_count": error or 0,
        "error_rate": round((error or 0) / total * 100, 2) if total else 0,
        "avg_latency_ms": round(avg_lat or 0, 1),
        "avg_retrieval_ms": round(avg_ret or 0, 1),
        "avg_tokens": round(avg_tok or 0, 1),
        "total_tokens": sum_tok or 0,
    }
