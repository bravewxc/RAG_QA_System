# -*- coding: utf-8 -*-
"""FastAPI 服务化：将 CombineClient 封装为 REST 接口。

接口：
- GET  /api/kb/list    获取知识库列表
- POST /api/chat        SSE 流式问答
- POST /api/upload      上传文件到指定知识库
- GET  /api/health      健康检查

启动：python -m app.api_server  或  uvicorn app.api_server:app --host 0.0.0.0 --port 8000
"""
import os
import shutil
import threading
import time
import uuid
from collections import defaultdict, deque

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel

from core.combine_client import CombineClient
from infra.models import ALI_TONGYI_MAX_MODEL, ALI_TONGYI_DEEPSEEK_V4_FLASH
from infra.observability import setup_langsmith_tracing, RequestStats, get_stats_summary
from infra.config import settings

app = FastAPI(title="通用文档问答 API", version="1.0.0")

# 允许跨域（便于 Java 后端 / 前端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 可观测性：启用 LangSmith 全链路追踪
setup_langsmith_tracing()

# ============ API 限流：滑动窗口算法 ============
_RATE_LIMIT = settings.API_RATE_LIMIT  # 如 "30/minute"
_RATE_MAX, _RATE_UNIT = _RATE_LIMIT.split("/")
_RATE_MAX = int(_RATE_MAX)
_RATE_WINDOW = 60 if _RATE_UNIT == "minute" else 1
_rate_records: dict = defaultdict(deque)
_rate_lock = threading.Lock()


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """API 限流中间件：按客户端IP做滑动窗口限流，超限返回429"""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    with _rate_lock:
        q = _rate_records[client_ip]
        # 清理过期记录
        while q and q[0] < now - _RATE_WINDOW:
            q.popleft()
        if len(q) >= _RATE_MAX:
            return JSONResponse(
                status_code=429,
                content={"detail": f"请求过于频繁，限流 {_RATE_MAX}/{_RATE_UNIT}，请稍后重试"},
            )
        q.append(now)
    return await call_next(request)


# 全局 CombineClient 实例（启动时初始化，预热知识库）
client = CombineClient()
client.warmup_indexes()

LLM_MODELS = [ALI_TONGYI_MAX_MODEL, ALI_TONGYI_DEEPSEEK_V4_FLASH]


class ChatRequest(BaseModel):
    question: str
    collection: str = ""
    model: str = ALI_TONGYI_DEEPSEEK_V4_FLASH
    max_length: int = 4000
    temperature: float = 0.7
    session_id: str = ""


@app.get("/api/health")
async def health():
    """健康检查"""
    return {"status": "ok", "timestamp": time.time()}


@app.get("/api/kb/list")
async def kb_list():
    """获取知识库列表"""
    kbs = client.load_knowledge()
    return {"knowledge_bases": kbs}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """SSE 流式问答接口

    返回 Server-Sent Events 流，每条事件格式：data: {"answer": "..."}\n\n
    """
    if not req.question:
        raise HTTPException(status_code=400, detail="question 不能为空")

    session_id = req.session_id or f"api_{uuid.uuid4().hex[:8]}"
    stats = RequestStats(session_id, req.question, req.collection, req.model)

    def event_stream():
        total_answer = []
        try:
            stats.mark_retrieval_start()
            for chunk in client.stream(
                req.question, req.collection, model=req.model,
                max_length=req.max_length, temperature=req.temperature,
                session_id=session_id,
            ):
                if "answer" in chunk and chunk["answer"]:
                    if not stats.retrieval_end:
                        stats.mark_retrieval_end()
                    total_answer.append(chunk["answer"])
                    yield f"data: {__import__('json').dumps({'answer': chunk['answer']}, ensure_ascii=False)}\n\n"
            stats.answer_length = len("".join(total_answer))
            yield f"data: {__import__('json').dumps({'done': True, 'session_id': session_id}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception(f"chat API 异常: {e}")
            stats.mark_error(str(e))
            yield f"data: {__import__('json').dumps({'error': str(e)}, ensure_ascii=False)}\n\n"
        finally:
            stats.save()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/stats")
async def stats(hours: int = 1):
    """获取最近N小时的统计摘要（Token消耗/检索延迟/错误率）"""
    return get_stats_summary(hours)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), collection: str = Form("")):
    """上传文件到指定知识库，自动触发索引重建"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    # 未指定知识库时用默认库
    if not collection:
        from core.knowledge import KNOWLEDGE_DIR
        dirs = sorted(d for d in os.listdir(KNOWLEDGE_DIR)
                      if os.path.isdir(os.path.join(KNOWLEDGE_DIR, d)))
        if not dirs:
            raise HTTPException(status_code=400, detail="尚无知识库目录，请先创建")
        collection = dirs[0]

    dest_dir = os.path.join("./chroma/knowledge", collection)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, file.filename)

    # 保存文件
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info(f"API 上传文件 [{file.filename}] 到知识库 [{collection}]")

    # 后台重建索引
    def _rebuild():
        try:
            from core.knowledge import get_md5, create_indexes, KNOWLEDGE_DIR
            from infra.models import get_ali_embeddings
            collection_md5 = get_md5(collection)
            built = create_indexes(collection_md5, dest_dir, get_ali_embeddings())
            if built:
                client._MyKnowledge__retrievers[collection_md5] = built
                logger.info(f"知识库 [{collection}] 索引重建完成")
        except Exception as e:
            logger.error(f"索引重建失败: {e}")

    threading.Thread(target=_rebuild, daemon=True).start()

    return JSONResponse({
        "status": "ok",
        "filename": file.filename,
        "collection": collection,
        "message": f"文件已上传至 [{collection}]，正在后台建立索引"
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
