# General Doc QA System（通用文档智能问答系统）

基于 RAG（检索增强生成）架构的企业级文档智能问答系统，以财务年报场景为切入点，提供多轮对话、跨期对比、跨知识库检索能力。系统经过 24 项迭代优化与完整工程化改造，RAGAS 四维评测指标从基线 0.49/0.40/0.56/0.36 提升至 **0.73/0.86/0.88/0.64**（context_precision / context_recall / faithfulness / answer_relevancy）。

## 功能特性

- **多格式文档解析**：支持 PDF / Word / TXT / Markdown，财报章节感知切分，保证表格与附注完整性
- **混合检索架构**：BM25（jieba 中文分词 + 领域词典）+ 密集检索双路召回，Rerank 精排，Small-to-Big 父子分层索引
- **多轮对话记忆**：SQLite 持久化历史，滑动窗口 + LLM 摘要压缩，跨知识库标签记忆
- **抗幻觉引导**：渐进式 Prompt 优化，明细未披露时诚实拒答而非编造
- **完整评测体系**：RAGAS 四维指标 + 127 条黄金评测集（五大能力维度、三级难度），两阶段分离流水线
- **生产级工程化**：统一重试、三级缓存（Embedding / 检索 / 语义）、FastAPI SSE 服务化、Docker 一键部署、LangSmith 全链路追踪、并发限流

## 架构

```
                        ┌────────────────────────────────┐
                        │           用户 / Java 后端       │
                        └───────┬───────────────┬────────┘
                                │               │
                    ┌───────────▼──┐   ┌────────▼─────────┐
                    │ Gradio 前端   │   │  FastAPI (SSE)   │
                    │  (演示用)     │   │  /api/chat 等     │
                    └───────────┬──┘   └────────┬─────────┘
                                │               │
                    ┌───────────▼───────────────▼─────────┐
                    │        app/  应用入口层               │
                    ├─────────────────────────────────────┤
                    │        core/  核心业务层              │
                    │  CombineClient /MyKnowledge /Loader │
                    ├─────────────────────────────────────┤
                    │        infra/  基础设施层             │
                    │  config / models / retry / cache    │
                    │  observability / logger             │
                    └──────────────┬──────────────────────┘
                                   │
        ┌──────────────┬───────────┼──────────────┬─────────────┐
        ▼              ▼           ▼              ▼             ▼
   ┌─────────┐  ┌──────────┐ ┌─────────┐  ┌──────────┐  ┌────────────┐
   │ Chroma  │  │  BM25    │ │ Rerank  │  │ DashScope│  │ SQLite     │
   │ 向量库   │  │ 稀疏检索   │ │ 精排    │  │ LLM/Embed│  │ 缓存/历史    │
   └─────────┘  └──────────┘ └─────────┘  └──────────┘  └────────────┘
```

**检索链路**：原始 query + RePhraseQuery 重写 → DualQuery 双路并行检索（各含 BM25+密集混合召回 + Small-to-Big 父子映射）→ 哈希去重合并 → 跨期分路补充 → Rerank 精排（top_n=4 + 元数据 soft boost）→ 结构化主表注入 → LLM 流式生成

## 技术栈

| 类别 | 技术 |
| --- | --- |
| 大模型 | deepseek-v4-flash（生成）、qwen3.7-flash（评测） |
| Embedding | text-embedding-v4（1024 维，cosine 距离） |
| 向量数据库 | Chroma（HNSW 索引，本地持久化） |
| 检索 | BM25（jieba + 40 个财务专有名词词典）、gte-rerank-v2、Small-to-Big、DualQuery |
| 框架 | LangChain、FastAPI、Gradio、pydantic-settings |
| 评估 | RAGAS 0.4.3 + 自研评测流水线 |
| 工程化 | Docker、tenacity（重试）、三级缓存、LangSmith、Semaphore 限流 |

## 快速开始

### 环境要求

- Python 3.12+
- 阿里云百炼 DashScope API Key（[获取地址](https://bailian.console.aliyun.com/)）

### 安装

```bash
# 1. 克隆项目
git clone https://github.com/your-username/General_Doc_QA_System_Optimize.git
cd General_Doc_QA_System_Optimize

# 2. 创建虚拟环境并安装依赖
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DASHSCOPE_API_KEY

# 4. 放置知识库文档（目录即知识库）
mkdir -p chroma/knowledge/报表库
# 将你的 PDF/Word/TXT 文档放入该目录，例如年报文件
```

> 说明：出于版权考虑，知识库源文档不随仓库分发。启动后系统会自动扫描 `chroma/knowledge/` 下的子目录并构建索引（懒加载 + 后台预热），无需手动执行入库命令。

### 启动

```bash
# 方式一：Gradio Web 界面（演示）
python -m app.main
# 浏览器访问 http://127.0.0.1:7860

# 方式二：FastAPI 服务（供后端集成）
python -m app.api_server
# 或
uvicorn app.api_server:app --host 0.0.0.0 --port 8000
# API 文档自动生成于 http://127.0.0.1:8000/docs
```

### Docker 部署

```bash
# 一键启动（应用 + Chroma + Redis）
docker-compose up -d
```

配置通过根目录 `.env` 文件注入，所有可配置项见 `.env.example`。

## REST API

| 接口 | 方法 | 说明 |
| --- | --- | --- |
| `/api/chat` | POST | SSE 流式问答，逐 chunk 推送 `{"answer": "..."}`，结束发 `{"done": true}` |
| `/api/upload` | POST | 上传文档到指定知识库（multipart/form-data），后台自动重建索引 |
| `/api/kb/list` | GET | 获取知识库列表 |
| `/api/stats` | GET | 获取统计摘要（Token 消耗 / 延迟 / 错误率） |
| `/api/health` | GET | 健康检查 |

**SSE 调用示例**（Python）：

```python
import requests

with requests.post(
    "http://127.0.0.1:8000/api/chat",
    json={"question": "2025年营业收入是多少", "collection": "报表库"},
    stream=True,
) as r:
    for line in r.iter_lines():
        if line.startswith(b"data: "):
            print(line[6:].decode("utf-8"))
```

Java 后端可通过 OkHttp / WebClient 的 SSE 客户端消费同一事件流。

## 评测

```bash
# 全量评测（检索 + 生成 + RAGAS）
python evaluation/eval_pipeline.py --run-name "my-eval-run"

# 改动仅涉及 Prompt / 生成参数时，复用检索缓存快速重评
python evaluation/eval_pipeline.py --run-name "my-eval-run" --skip-invoke
```

- **评测集**：`evaluation/评测集/报表库/golden_set.jsonl`，127 条，覆盖检索定位（21%）、数值精确性（25%）、推理与分析（35%）、口径一致性校验（12%）、抗幻觉与拒答（7%）五个维度，hard 题占比 ≥30%
- **输出**：每次运行生成 CSV 落库（`evaluation/results/eval_runs.csv`）+ Markdown 对比报告（`evaluation/reports/`），自动记录环境指纹（chunk 参数 / 模型 / Prompt 哈希）供横向对比
- **评测与线上同源**：评测流水线与线上服务共用 `_prepare_context` 与 Prompt 模板，保证"评测即线上"

**最终指标**（127 条，比亚迪 2023-2025 年报）：

| 指标 | 基线 | 最终 | 变化 |
| --- | --- | --- | --- |
| context_precision (CP) | 0.492 | 0.730 | +48.4% |
| context_recall (CR) | 0.404 | 0.862 | +113.4% |
| faithfulness (F) | 0.560 | 0.880 | +57.1% |
| answer_relevancy (AR) | 0.360 | 0.640 | +77.8% |

## 项目结构

```
.
├── app/                        # 应用入口层
│   ├── main.py               # Gradio Web 界面
│   └── api_server.py         # FastAPI REST/SSE 服务
├── core/                       # 核心业务层
│   ├── combine_client.py     # 问答客户端（检索编排 + 记忆管理 + 缓存接入）
│   ├── knowledge.py          # 知识库管理（懒加载 + 预热 + 并发锁）
│   └── custom_loader.py      # 文档加载与切分（财报章节感知 + 父子 chunk）
├── infra/                      # 基础设施层
│   ├── config.py             # pydantic-settings 全局配置
│   ├── models.py             # 模型客户端（并发控制 + 重试）
│   ├── retry_utils.py        # tenacity 统一重试（429/5xx 指数退避）
│   ├── cache_utils.py        # 三级缓存（Embedding/检索/语义）
│   ├── observability.py      # LangSmith 追踪 + 会话统计
│   └── logger.py             # loguru 日志配置
├── evaluation/                 # 评测体系
│   ├── eval_pipeline.py      # 两阶段分离评测流水线
│   └── 评测集/报表库/          # 127 条黄金评测集
├── structured_data/            # 三大主表结构化 JSON（注入生成阶段）
├── chroma/
│   └── knowledge/             # 知识库源文档（自行放置，目录即知识库）
├── Dockerfile
├── docker-compose.yml         # app + chromadb + redis
├── requirements.txt
└── .env.example               # 配置模板
```

## 配置说明

所有配置通过 `.env` 管理（复制 `.env.example` 后编辑），关键项：

| 变量 | 说明 | 默认值 |
| --- | --- | --- |
| `DASHSCOPE_API_KEY` | 阿里云百炼 API 密钥（必填） | — |
| `ANSWER_MODEL` | 答案生成模型 | deepseek-v4-flash-0731 |
| `EMBEDDING_MODEL` | 向量模型 | text-embedding-v4 |
| `RERANK_MODEL` | 重排序模型 | gte-rerank-v2 |
| `DASHSCOPE_MAX_CONCURRENCY` | LLM 并发上限 | 8 |
| `API_RATE_LIMIT` | API 限流 | 30/minute |
| `SEMANTIC_CACHE_THRESHOLD` | 语义缓存相似度阈值 | 0.92 |
| `LANGSMITH_TRACING` | LangSmith 全链路追踪开关 | false |

## 可靠性设计

- **统一重试**：LLM / Embedding / Rerank 三类调用统一 tenacity 装饰器，429 与 5xx 指数退避重试 3 次（1s/2s/4s），其余 4xx 快速失败
- **三级缓存**：Embedding 缓存（query 哈希 → 向量）→ 检索缓存（(query, kb) → 文档列表，TTL 1h）→ 语义缓存（相似度 > 0.92 复用历史答案，TTL 2h），重复问题延迟趋近 0
- **并发控制**：Semaphore 限制 DashScope 并发防止 RPM 限流；API 层滑动窗口限流防恶意调用
- **全局兜底**：生成流程 try/except 包裹，异常时输出友好提示并记录日志
- **可观测性**：LangSmith 追踪检索/生成全链路；SQLite 落库每次会话的 Token 消耗、检索延迟、错误率

## License

MIT
