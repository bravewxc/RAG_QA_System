FROM python:3.12-slim

WORKDIR /app

# 系统依赖（PDF解析、中文分词等需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先拷贝依赖文件，利用 Docker 层缓存
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/chroma/knowledge /app/chroma/vector_store /app/db/cache /app/db/chat_history /app/logs

EXPOSE 8000 7860

CMD ["python", "-m", "app.api_server"]
