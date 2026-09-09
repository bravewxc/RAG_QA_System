# -*- coding: utf-8 -*-
"""
RAG 评测流水线：读取评测集 -> 批量检索 -> 批量回答 -> Ragas 四指标 -> 对比报告 -> CSV/DB 记录

两阶段分离设计（检索 / 回答）：
    - 阶段一【检索】：逐条调用 CombineClient.retrieve，产出 contexts（检索产物），
      缓存到 latest_invoke.json，供后续运行复用；
    - 阶段二【回答】：逐条调用 CombineClient.answer_with_context，基于缓存的 contexts
      生成 answer（生成与 Prompt/模型相关，每次运行始终重新生成）。

用法：
    python eval_pipeline.py                          # 全量评测（默认配置）
    python eval_pipeline.py --limit 20               # 只评测前 20 条（快速验证）
    python eval_pipeline.py --testset 评测集/报表库/golden_set.jsonl
    python eval_pipeline.py --collection 报表库
    python eval_pipeline.py --model deepseek-v4-flash-0731
    python eval_pipeline.py --run-name "基线-20260803"
    python eval_pipeline.py --skip-invoke            # 复用检索缓存（latest_invoke.json），跳过检索，只重新生成回答

改动与重跑规则：
    - 改检索（Chunk 策略/Embedding/检索参数）：删 latest_invoke.json 重跑（重新检索+回答）
    - 改 Prompt/模型：--skip-invoke 复用检索缓存，只重新生成回答，快得多

每次运行都会：
    1. 逐条检索知识库并生成回答（独立 session_id 隔离历史）
    2. Ragas 计算 context_precision / context_recall / faithfulness / answer_relevancy
    3. 输出对比报告 evaluation/reports/eval_report_<时间戳>.md
    4. 将「时间-Chunk策略-Embedding-Prompt-四指标」写入 evaluation/results/eval_runs.csv 与 eval_runs.db
"""
import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# 将项目根目录加入 sys.path（pipeline 位于 ./evaluation/ 下）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.documents import Document  # noqa: E402
from datasets import Dataset  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas import evaluate  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

from core.combine_client import CombineClient  # noqa: E402
from infra.models import ALI_TONGYI_MAX_MODEL, ALI_TONGYI_DEEPSEEK_V4_FLASH, ALI_TONGYI_EMBEDDING_MODEL, get_ali_embeddings, get_lc_model_client  # noqa: E402

# ============ 路径配置 ============
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TESTSET = os.path.join(EVAL_DIR, '评测集', '报表库', 'golden_set.jsonl')
REPORT_DIR = os.path.join(EVAL_DIR, 'reports')
RESULT_DIR = os.path.join(EVAL_DIR, 'results')
CSV_FILE = os.path.join(RESULT_DIR, 'eval_runs.csv')
DB_FILE = os.path.join(RESULT_DIR, 'eval_runs.db')
RUNS_JSON = os.path.join(RESULT_DIR, 'latest_invoke.json')  # 最近一次 invoke 结果缓存

# 评测用固定参数（保证可复现，与生产 Gradio 默认参数区分）
ANSWER_MODEL = ALI_TONGYI_DEEPSEEK_V4_FLASH  # 默认评测模型（更稳、QPS 高）
EVAL_MODEL = ALI_TONGYI_MAX_MODEL
EVAL_MAX_LENGTH = 1024  # AR优化建议1：从512提升至1024，避免长答案（推理/跨期题）被max_tokens截断导致AR计0
EVAL_TEMPERATURE = 0.1

# 方案A：并发化配置
# retrieve 阶段不涉及 SQLite，可高并发；answer 阶段写 SQLite 历史，用较低并发 + timeout=30 防锁冲突
RETRIEVE_WORKERS = 8
ANSWER_WORKERS = 5
_print_lock = threading.Lock()


# ============ 运行环境指纹（用于对比 CSV/DB） ============
def get_chunk_strategy():
    """从 custom_loader 源文件读取当前切分策略（chunk_size/chunk_overlap）"""
    loader_src_path = os.path.join(PROJECT_ROOT, 'core', 'custom_loader.py')
    with open(loader_src_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chunk_size = None
    chunk_overlap = None
    for line in text.splitlines():
        line = line.strip()
        if 'chunk_size' in line and '=' in line:
            chunk_size = line.split('=')[-1].strip().rstrip(',')
        if 'chunk_overlap' in line and '=' in line:
            chunk_overlap = line.split('=')[-1].strip().rstrip(',')
    return f"chunk_size={chunk_size},chunk_overlap={chunk_overlap}"


def get_embedding_name():
    return ALI_TONGYI_EMBEDDING_MODEL


def get_prompt_fingerprint():
    """从 combine_client 的 qa_system_prompt 计算摘要，作为 Prompt 版本指纹"""
    from core.combine_client import qa_system_prompt
    return qa_system_prompt.strip().replace('\n', ' ')[:60] + '...'


# ============ 数据读写 ============
def load_testset(path, limit=None):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    return rows


def save_retrieve_cache(results):
    """保存检索产物（contexts，不含 answer）到 latest_invoke.json"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(RUNS_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def load_retrieve_cache():
    """读取最近一次检索缓存；兼容旧版缓存（contexts 为纯字符串列表）"""
    if os.path.exists(RUNS_JSON):
        with open(RUNS_JSON, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def to_doc_list(contexts):
    """把缓存里的 contexts（dict 列表）还原为 Document 列表；兼容旧版纯字符串列表"""
    docs = []
    for c in contexts:
        if isinstance(c, dict):
            docs.append(Document(page_content=c.get('page_content', ''), metadata=c.get('metadata', {})))
        else:  # 旧版缓存：直接是 page_content 字符串
            docs.append(Document(page_content=str(c), metadata={}))
    return docs


def contexts_to_strs(contexts):
    """把 contexts（dict 或 str）转成 page_content 字符串列表，供 Ragas 使用"""
    return [c['page_content'] if isinstance(c, dict) else str(c) for c in contexts]


# ============ 阶段一：批量检索（产物可缓存，并发执行） ============
def batch_retrieve(client, rows, collection, max_workers=RETRIEVE_WORKERS):
    """
    并发检索知识库，只记录 contexts（检索产物，不含 answer）。
    contexts 以 {page_content, metadata} 结构序列化保存，可还原为 Document 供回答阶段使用。
    retrieve 不涉及 SQLite，线程安全，可高并发。
    """
    total = len(rows)
    results = [None] * total

    def _retrieve_one(args):
        idx, item = args
        q = item['question']
        docs = client.retrieve(q, collection)
        return idx, {
            "id": item.get("id", f"Q{idx}"),
            "question": q,
            "ground_truth": item.get("ground_truth", ""),
            "contexts": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            "category": item.get("category", ""),
            "type": item.get("type", ""),
            "difficulty": item.get("difficulty", ""),
            "gt_type": item.get("gt_type", "answerable"),
        }

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_retrieve_one, (i, item)) for i, item in enumerate(rows, 1)]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx - 1] = result
            done += 1
            with _print_lock:
                print(f"[{done}/{total}] {result['id']} | ctx={len(result['contexts'])}")
    return results


# ============ 阶段二：批量回答（每次运行重新生成，并发执行） ============
def batch_answer(client, rows, collection, max_workers=ANSWER_WORKERS):
    """
    基于已检索的 contexts 并发生成 answer。关键点：
    - 每条使用独立 session_id，回答前后 clear_history，避免上一题污染下一题上下文
    - 最多重试 2 次（模型偶发返回空答案/限流时提升稳定性）
    - 写 SQLite 历史用较低并发（5），配合 timeout=30 防锁冲突
    """
    total = len(rows)
    results = [None] * total

    def _answer_one(args):
        idx, item = args
        q = item['question']
        context_docs = to_doc_list(item['contexts'])
        session_id = f"eval_{int(time.time())}_{idx}"
        client.clear_history(session_id)
        answer = ''
        for attempt in range(3):
            try:
                answer = client.answer_with_context(
                    q, context_docs, collection,
                    model=ANSWER_MODEL, max_length=EVAL_MAX_LENGTH,
                    temperature=EVAL_TEMPERATURE, session_id=session_id,
                    request_timeout=180,  # AR优化建议2：长上下文hard题生成超时从120s提到180s，减少空答案
                )
                if answer.strip():
                    break
                with _print_lock:
                    print(f"[{idx}/{total}] 第{attempt + 1}次为空答案，重试...")
            except Exception as e:
                with _print_lock:
                    print(f"[{idx}/{total}] 第{attempt + 1}次异常: {q[:40]}... -> {e}")
            client.clear_history(session_id)
        client.clear_history(session_id)  # 评测后清空，防止 SQLite 历史膨胀
        return idx, {**item, "answer": answer}

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_answer_one, (i, item)) for i, item in enumerate(rows, 1)]
        for future in as_completed(futures):
            idx, result = future.result()
            results[idx - 1] = result
            done += 1
            with _print_lock:
                print(f"[{done}/{total}] {result['id']} | ans_len={len(result['answer'])}")
    return results


# ============ Ragas 评测 ============
def run_ragas(results):
    """组装 Dataset 并计算四指标，返回 (总体结果, 按类别/难度拆分)"""
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [contexts_to_strs(r["contexts"]) for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    })
    # request_timeout 从默认120提升至180：CP的prompt最复杂（含question+ground_truth+全部contexts，
    # 要求对每个chunk输出JSON verdict），qwen3.7-flash-2026-07-15作为思考类模型处理耗时较长，
    # 120s超时导致CP 100% nan。提升至180s给思考模型足够时间完成CP的复杂推理。
    llm = get_lc_model_client(model=EVAL_MODEL, request_timeout=180)
    embedding = get_ali_embeddings()
    # bypass_n=True：DashScope 思考类模型（qwen/deepseek enable_thinking）要求 n=1，
    # 而 Ragas 默认按 n>1 调用会报 "<400> The n parameter must be 1 when enable_thinking is true"
    vllm = LangchainLLMWrapper(llm, bypass_n=True)
    vllm_e = LangchainEmbeddingsWrapper(embedding)

    # max_workers 从默认16降至8：降低并发缓解 QPS 限流，减少因限流导致的 TimeoutError
    run_config = RunConfig(max_workers=8)

    result = evaluate(
        dataset,
        llm=vllm,
        embeddings=vllm_e,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        run_config=run_config,
    )
    df = result.to_pandas()
    return df, result


# ============ 对比报告 ============
def format_pct(v):
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "N/A"


def build_report(run_info, df, results):
    """生成 Markdown 对比报告"""
    lines = []
    lines.append(f"# RAG 评测报告 - {run_info['run_name']}")
    lines.append("")
    lines.append(f"- **评测时间**: {run_info['timestamp']}")
    lines.append(f"- **评测集**: {os.path.basename(run_info['testset'])}（{len(results)} 条）")
    lines.append(f"- **知识库**: {run_info['collection']}")
    lines.append(f"- **模型**: {run_info['model']}")
    lines.append(f"- **Chunk 策略**: {run_info['chunk_strategy']}")
    lines.append(f"- **Embedding**: {run_info['embedding']}")
    lines.append(f"- **Prompt**: {run_info['prompt_fingerprint']}")
    lines.append("")

    # 总体指标
    lines.append("## 一、总体指标")
    lines.append("")
    lines.append("| 指标 | 得分 |")
    lines.append("| --- | --- |")
    for col in ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']:
        if col in df.columns:
            lines.append(f"| {col} | {format_pct(df[col].mean())} |")
    lines.append("")

    # 按类别
    lines.append("## 二、按类别指标")
    lines.append("")
    lines.append("| 类别 | 条数 | context_precision | context_recall | faithfulness | answer_relevancy |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for cat in sorted(set(r['category'] for r in results)):
        idxs = [i for i, r in enumerate(results) if r['category'] == cat]
        sub = df.iloc[idxs]
        rows_ = [
            cat, str(len(idxs)),
            format_pct(sub['context_precision'].mean()) if 'context_precision' in sub else 'N/A',
            format_pct(sub['context_recall'].mean()) if 'context_recall' in sub else 'N/A',
            format_pct(sub['faithfulness'].mean()) if 'faithfulness' in sub else 'N/A',
            format_pct(sub['answer_relevancy'].mean()) if 'answer_relevancy' in sub else 'N/A',
        ]
        lines.append("| " + " | ".join(rows_) + " |")
    lines.append("")

    # 按难度
    lines.append("## 三、按难度指标")
    lines.append("")
    lines.append("| 难度 | 条数 | context_precision | context_recall | faithfulness | answer_relevancy |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for diff in sorted(set(r['difficulty'] for r in results)):
        idxs = [i for i, r in enumerate(results) if r['difficulty'] == diff]
        sub = df.iloc[idxs]
        rows_ = [
            diff, str(len(idxs)),
            format_pct(sub['context_precision'].mean()) if 'context_precision' in sub else 'N/A',
            format_pct(sub['context_recall'].mean()) if 'context_recall' in sub else 'N/A',
            format_pct(sub['faithfulness'].mean()) if 'faithfulness' in sub else 'N/A',
            format_pct(sub['answer_relevancy'].mean()) if 'answer_relevancy' in sub else 'N/A',
        ]
        lines.append("| " + " | ".join(rows_) + " |")
    lines.append("")

    # 任务3：按 gt_type 分组统计（不可答题的 CR 无意义，标注 N/A）
    lines.append("## 四、按题目类型（gt_type）指标")
    lines.append("")
    lines.append('> 不可答题（unanswerable）的 GT 是"未披露"，检索端找不到证据，CR 恒为 0 无意义，标注 N/A 不计入 CR 均值。')
    lines.append("")
    lines.append("| 类型 | 条数 | context_precision | context_recall | faithfulness | answer_relevancy |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for gt in ['answerable', 'unanswerable']:
        idxs = [i for i, r in enumerate(results) if r.get('gt_type', 'answerable') == gt]
        if not idxs:
            continue
        sub = df.iloc[idxs]
        cr_val = format_pct(sub['context_recall'].mean()) if gt == 'answerable' else 'N/A(不可答题)'
        rows_ = [
            gt, str(len(idxs)),
            format_pct(sub['context_precision'].mean()) if 'context_precision' in sub else 'N/A',
            cr_val,
            format_pct(sub['faithfulness'].mean()) if 'faithfulness' in sub else 'N/A',
            format_pct(sub['answer_relevancy'].mean()) if 'answer_relevancy' in sub else 'N/A',
        ]
        lines.append("| " + " | ".join(rows_) + " |")
    lines.append("")

    # 历次运行对比（从 DB 读）
    lines.append("## 五、历次运行对比（来自 eval_runs.db）")
    lines.append("")
    prev = list_prev_runs(exclude=run_info['timestamp'])
    if prev:
        lines.append("| 时间 | 运行名 | Chunk策略 | Embedding | Prompt | CP | CR | F | AR |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        cur_row = {
            'timestamp': run_info['timestamp'], 'run_name': run_info['run_name'],
            'chunk_strategy': run_info['chunk_strategy'], 'embedding': run_info['embedding'],
            'prompt_fingerprint': run_info['prompt_fingerprint'],
            'context_precision': format_pct(df['context_precision'].mean()),
            'context_recall': format_pct(df['context_recall'].mean()),
            'faithfulness': format_pct(df['faithfulness'].mean()),
            'answer_relevancy': format_pct(df['answer_relevancy'].mean()),
        }
        lines.append(("| {timestamp} | {run_name} | {chunk_strategy} | {embedding} | {prompt_fingerprint} | "
                      "{context_precision} | {context_recall} | {faithfulness} | {answer_relevancy} |").format(**cur_row))
        for r in prev:
            lines.append(("| {timestamp} | {run_name} | {chunk_strategy} | {embedding} | {prompt_fingerprint} | "
                          "{context_precision} | {context_recall} | {faithfulness} | {answer_relevancy} |").format(**r))
    else:
        lines.append("（首次运行，暂无历史对比）")
    lines.append("")

    # 明细表（截断上下文）
    lines.append("## 六、逐题明细")
    lines.append("")
    lines.append("| ID | 类别 | 难度 | gt_type | CP | CR | F | AR | 问题 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(results):
        cp = format_pct(df.iloc[i]['context_precision']) if 'context_precision' in df.columns else 'N/A'
        cr = format_pct(df.iloc[i]['context_recall']) if 'context_recall' in df.columns else 'N/A'
        f_ = format_pct(df.iloc[i]['faithfulness']) if 'faithfulness' in df.columns else 'N/A'
        ar = format_pct(df.iloc[i]['answer_relevancy']) if 'answer_relevancy' in df.columns else 'N/A'
        q = r['question'].replace('|', '\\|')
        lines.append(f"| {r['id']} | {r['category']} | {r['difficulty']} | {r.get('gt_type', 'answerable')} | {cp} | {cr} | {f_} | {ar} | {q[:40]} |")
    lines.append("")
    return "\n".join(lines)


# ============ CSV / DB 记录 ============
def init_db():
    os.makedirs(RESULT_DIR, exist_ok=True)
    con = sqlite3.connect(DB_FILE)
    con.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            run_name TEXT,
            collection TEXT,
            model TEXT,
            chunk_strategy TEXT,
            embedding TEXT,
            prompt_fingerprint TEXT,
            context_precision REAL,
            context_recall REAL,
            faithfulness REAL,
            answer_relevancy REAL
        )
    """)
    con.commit()
    return con


def write_csv(run_info, scores):
    os.makedirs(RESULT_DIR, exist_ok=True)
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestamp', 'run_name', 'collection', 'model',
            'chunk_strategy', 'embedding', 'prompt_fingerprint',
            'context_precision', 'context_recall', 'faithfulness', 'answer_relevancy',
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'timestamp': run_info['timestamp'],
            'run_name': run_info['run_name'],
            'collection': run_info['collection'],
            'model': run_info['model'],
            'chunk_strategy': run_info['chunk_strategy'],
            'embedding': run_info['embedding'],
            'prompt_fingerprint': run_info['prompt_fingerprint'],
            **scores,
        })


def write_db(con, run_info, scores):
    con.execute("""
        INSERT INTO eval_runs
        (timestamp, run_name, collection, model, chunk_strategy, embedding, prompt_fingerprint,
         context_precision, context_recall, faithfulness, answer_relevancy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_info['timestamp'], run_info['run_name'], run_info['collection'], run_info['model'],
        run_info['chunk_strategy'], run_info['embedding'], run_info['prompt_fingerprint'],
        scores['context_precision'], scores['context_recall'], scores['faithfulness'], scores['answer_relevancy'],
    ))
    con.commit()


def list_prev_runs(exclude=None):
    """读取 DB 中历次运行记录（排除当前这次），供对比报告使用"""
    if not os.path.exists(DB_FILE):
        return []
    try:
        con = sqlite3.connect(DB_FILE)
        con.row_factory = sqlite3.Row
        if exclude:
            cur = con.execute("SELECT * FROM eval_runs WHERE timestamp != ? ORDER BY id", (exclude,))
        else:
            cur = con.execute("SELECT * FROM eval_runs ORDER BY id")
        rows = [dict(r) for r in cur.fetchall()]
        con.close()
        return rows
    except Exception:
        return []


# ============ 主流程 ============
def main():
    parser = argparse.ArgumentParser(description='RAG 评测流水线')
    parser.add_argument('--testset', default=DEFAULT_TESTSET, help='评测集 JSONL 路径')
    parser.add_argument('--collection', default='报表库', help='知识库名称')
    parser.add_argument('--limit', type=int, default=None, help='只评测前 N 条')
    parser.add_argument('--model', default=EVAL_MODEL, help='评测用 LLM 模型')
    parser.add_argument('--run-name', default=None, help='本次运行名称（用于对比标识）')
    parser.add_argument('--skip-invoke', action='store_true', help='复用检索缓存（latest_invoke.json），跳过检索，只重新生成回答')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    run_name = args.run_name or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(REPORT_DIR, exist_ok=True)

    # 1. 加载评测集
    rows = load_testset(args.testset, args.limit)
    print(f"加载评测集: {len(rows)} 条 <- {args.testset}")
    if not rows:
        print("评测集为空，退出")
        return

    # 2. 阶段一：批量检索（产物可缓存复用）
    client = CombineClient()
    if args.skip_invoke:
        cached = load_retrieve_cache()
        if cached is None:
            print("未找到检索缓存（latest_invoke.json），请去掉 --skip-invoke 重新跑")
            return
        results = cached
        print(f"复用检索缓存: {len(results)} 条（跳过检索，重新生成回答）")
    else:
        # 预热知识库索引（首次会建索引）
        print(f"准备知识库 [{args.collection}] 检索器...")
        client.get_retrievers(args.collection)
        results = batch_retrieve(client, rows, args.collection)
        save_retrieve_cache(results)

    # 3. 阶段二：批量回答（始终重新生成，answer 与 Prompt/模型相关，不能跨配置复用）
    results = batch_answer(client, results, args.collection)

    # 4. Ragas 四指标
    df, ragas_result = run_ragas(results)

    # 任务3：按 gt_type 分别统计指标
    # 不可答题（gt_type=unanswerable）的 CR 无意义（GT是"未披露"，检索端找不到证据恒为0），
    # 排除在 CR 均值之外；CP/F/AR 仍按全部题目计算（保留检索精度与生成质量评估）。
    # gt_type 从评测集透传到 results，再映射到 df 行。
    gt_types = [r.get('gt_type', 'answerable') for r in results]
    answerable_mask = [gt == 'answerable' for gt in gt_types]
    unanswerable_mask = [gt == 'unanswerable' for gt in gt_types]
    df_answerable = df.iloc[[i for i, m in enumerate(answerable_mask) if m]]
    df_unanswerable = df.iloc[[i for i, m in enumerate(unanswerable_mask) if m]]

    scores = {
        'context_precision': float(df['context_precision'].mean()),
        'context_recall': float(df_answerable['context_recall'].mean()),  # CR 仅对可答题计算
        'faithfulness': float(df['faithfulness'].mean()),
        'answer_relevancy': float(df['answer_relevancy'].mean()),
    }
    print(f"\n四指标(可答题CR): CP={scores['context_precision']:.4f} CR={scores['context_recall']:.4f} "
          f"F={scores['faithfulness']:.4f} AR={scores['answer_relevancy']:.4f}")
    # 全量 CR（含不可答题）作为参考对照
    cr_all = float(df['context_recall'].mean())
    print(f"参考-全量CR(含不可答题): {cr_all:.4f}")
    if len(df_unanswerable) > 0:
        print(f"不可答题({len(df_unanswerable)}条): F={df_unanswerable['faithfulness'].mean():.4f} "
              f"AR={df_unanswerable['answer_relevancy'].mean():.4f}")
    if len(df_answerable) > 0:
        print(f"可答题({len(df_answerable)}条): CP={df_answerable['context_precision'].mean():.4f} "
              f"CR={df_answerable['context_recall'].mean():.4f} "
              f"F={df_answerable['faithfulness'].mean():.4f} "
              f"AR={df_answerable['answer_relevancy'].mean():.4f}")

    # 5. 运行环境指纹
    run_info = {
        'timestamp': timestamp,
        'run_name': run_name,
        'collection': args.collection,
        'model': args.model,
        'chunk_strategy': get_chunk_strategy(),
        'embedding': get_embedding_name(),
        'prompt_fingerprint': get_prompt_fingerprint(),
        'testset': args.testset,
    }

    # 6. 对比报告
    report = build_report(run_info, df, results)
    report_file = os.path.join(REPORT_DIR, f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"对比报告: {report_file}")

    # 7. 写入 CSV/DB
    write_csv(run_info, scores)
    con = init_db()
    write_db(con, run_info, scores)
    print(f"已记录到 CSV: {CSV_FILE}")
    print(f"已记录到 DB: {DB_FILE}")


if __name__ == '__main__':
    main()
