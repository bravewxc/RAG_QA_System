# -*- coding: utf-8 -*-
"""多级缓存：Embedding缓存 + 检索缓存 + 语义缓存。

三级缓存策略：
1. Embedding缓存：query文本 → 向量（SQLite存储，hash去重）
   - 相同query的embedding不重复调用API
2. 检索缓存：(query, kb) → 文档列表（SQLite存储，TTL 1小时）
   - 相同问题不重复检索知识库
3. 语义缓存：相似度>0.92 复用历史答案（内存+向量比对）
   - 语义相近的问题直接返回历史答案

存储：统一用SQLite（项目已有依赖，无需引入Redis）
"""
import hashlib
import json
import sqlite3
import threading
import time
from typing import List, Optional

import numpy as np
from langchain_core.documents import Document
from loguru import logger

# 缓存数据库路径
CACHE_DB_DIR = "./db/cache"
CACHE_DB_PATH = f"{CACHE_DB_DIR}/rag_cache.db"
_EMBEDDING_CACHE: dict = {}  # 进程内embedding缓存（query_hash → vector）
_CACHE_LOCK = threading.Lock()
_cache_conn = None


def _get_cache_conn():
    """获取缓存数据库连接（线程安全的单例）"""
    global _cache_conn
    if _cache_conn is None:
        import os
        os.makedirs(CACHE_DB_DIR, exist_ok=True)
        _cache_conn = sqlite3.connect(CACHE_DB_PATH, timeout=30, check_same_thread=False)
        _cache_conn.execute("PRAGMA journal_mode=WAL")
        _init_tables(_cache_conn)
    return _cache_conn


def _init_tables(conn):
    """初始化缓存表"""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS retrieval_cache (
            query_hash TEXT,
            kb_name TEXT,
            query_text TEXT,
            contexts_json TEXT,
            created_at REAL,
            PRIMARY KEY (query_hash, kb_name)
        );
        CREATE TABLE IF NOT EXISTS semantic_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT,
            query_hash TEXT,
            answer TEXT,
            query_embedding_json TEXT,
            created_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_retrieval ON retrieval_cache(query_hash, kb_name);
    """)
    conn.commit()


def _hash_query(query: str) -> str:
    """对query做hash，作为缓存key"""
    return hashlib.md5(query.strip().encode("utf-8")).hexdigest()


# ============ 第一级：Embedding缓存 ============
def get_cached_embedding(query: str) -> Optional[List[float]]:
    """从进程内缓存获取embedding"""
    qh = _hash_query(query)
    return _EMBEDDING_CACHE.get(qh)


def set_cached_embedding(query: str, vector: List[float]):
    """将embedding写入进程内缓存"""
    qh = _hash_query(query)
    _EMBEDDING_CACHE[qh] = vector


# ============ 第二级：检索缓存 ============
RETRIEVAL_CACHE_TTL = 3600  # 检索缓存TTL：1小时


def get_cached_retrieval(query: str, kb_name: str) -> Optional[List[Document]]:
    """从SQLite获取检索缓存"""
    qh = _hash_query(query)
    with _CACHE_LOCK:
        conn = _get_cache_conn()
        row = conn.execute(
            "SELECT contexts_json, created_at FROM retrieval_cache WHERE query_hash=? AND kb_name=?",
            (qh, kb_name),
        ).fetchone()
    if row is None:
        return None
    contexts_json, created_at = row
    if time.time() - created_at > RETRIEVAL_CACHE_TTL:
        return None  # 过期
    try:
        contexts = json.loads(contexts_json)
        return [Document(page_content=c["page_content"], metadata=c.get("metadata", {})) for c in contexts]
    except Exception as e:
        logger.warning(f"检索缓存反序列化失败: {e}")
        return None


def set_cached_retrieval(query: str, kb_name: str, docs: List[Document]):
    """将检索结果写入SQLite缓存"""
    qh = _hash_query(query)
    contexts = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    with _CACHE_LOCK:
        conn = _get_cache_conn()
        conn.execute(
            "INSERT OR REPLACE INTO retrieval_cache (query_hash, kb_name, query_text, contexts_json, created_at) VALUES (?,?,?,?,?)",
            (qh, kb_name, query.strip(), json.dumps(contexts, ensure_ascii=False), time.time()),
        )
        conn.commit()


# ============ 第三级：语义缓存 ============
SEMANTIC_CACHE_THRESHOLD = 0.92  # 相似度阈值
SEMANTIC_CACHE_TTL = 7200  # 语义缓存TTL：2小时
MAX_SEMANTIC_ENTRIES = 200  # 最多保留200条语义缓存


def get_semantic_cache(query: str, query_embedding: List[float]) -> Optional[str]:
    """语义缓存查找：找到相似度>阈值的历史答案则返回"""
    with _CACHE_LOCK:
        conn = _get_cache_conn()
        rows = conn.execute(
            "SELECT answer, query_embedding_json FROM semantic_cache WHERE created_at > ? ORDER BY created_at DESC LIMIT ?",
            (time.time() - SEMANTIC_CACHE_TTL, MAX_SEMANTIC_ENTRIES),
        ).fetchall()
    if not rows:
        return None
    q_vec = np.array(query_embedding)
    best_sim = 0.0
    best_answer = None
    for answer, emb_json in rows:
        try:
            cached_vec = np.array(json.loads(emb_json))
            sim = float(np.dot(q_vec, cached_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(cached_vec) + 1e-8))
            if sim > best_sim:
                best_sim = sim
                best_answer = answer
        except Exception:
            continue
    if best_sim >= SEMANTIC_CACHE_THRESHOLD and best_answer:
        logger.info(f"语义缓存命中: sim={best_sim:.4f} >= {SEMANTIC_CACHE_THRESHOLD}")
        return best_answer
    return None


def set_semantic_cache(query: str, answer: str, query_embedding: List[float]):
    """将问答对写入语义缓存"""
    if not answer or not query_embedding:
        return
    qh = _hash_query(query)
    with _CACHE_LOCK:
        conn = _get_cache_conn()
        conn.execute(
            "INSERT INTO semantic_cache (query_text, query_hash, answer, query_embedding_json, created_at) VALUES (?,?,?,?,?)",
            (query.strip(), qh, answer, json.dumps(query_embedding), time.time()),
        )
        # 清理过期缓存
        conn.execute("DELETE FROM semantic_cache WHERE created_at < ?", (time.time() - SEMANTIC_CACHE_TTL,))
        conn.commit()
