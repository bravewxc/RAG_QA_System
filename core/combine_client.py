import json
import os
import re
from typing import Iterable, List, Optional, Tuple

import openai
import tiktoken
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import AddableDict
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import SQLChatMessageHistory
from loguru import logger

from core.knowledge import MyKnowledge
from infra.models import ALI_TONGYI_MAX_MODEL, get_lc_model_client, get_ali_embeddings
from infra.retry_utils import with_retry
from infra import cache_utils

# 会话历史 SQLite 存储配置：db/chat_history/chat_history.db
HISTORY_DB_DIR = "./db/chat_history"
HISTORY_DB_URL = "sqlite:///db/chat_history/chat_history.db?timeout=30"

# ============ 跨知识库对话记忆 配置 ============
# 知识库来源标签前缀：存储到历史消息中，标记每轮对话来自哪个知识库
KNOWLEDGE_SCOPE_PREFIX = "[知识库:"
KNOWLEDGE_SCOPE_SUFFIX = "]"
# 触发跨知识库对比检索的关键词（问题中包含这些词且历史出现多个知识库时）
COMPARISON_KEYWORDS = ("对比", "比较", "区别", "差异", "不同", "相比", "汇总", "梳理")

# ============ 滑动窗口 + 摘要压缩 配置 ============
# 各模型的上下文窗口上限（token）。用于 token 预算控制：
# 当历史对话 token 数超过 上限*COMPRESS_THRESHOLD_RATIO 时，触发摘要压缩。
MODEL_CONTEXT_LIMITS = {
    "qwen3.7-max-preview": 65536,
    "deepseek-v4-flash-0731": 384000,
}
DEFAULT_CONTEXT_LIMIT = 65536  # 未配置模型的默认上下文上限
COMPRESS_THRESHOLD_RATIO = 0.7 # token 预算占用比，超过 70% 触发压缩
KEEP_RECENT_ROUNDS = 6         # 压缩后保留最近 6 轮原文（一轮 = 一问一答 = 2条消息）
SUMMARY_MAX_TOKENS = 800       # 摘要输出的最大 token 数

# tiktoken 编码器（用于 token 估算，cl100k_base 对中文按字符级近似估算）
_token_encoder = None


def _get_token_encoder():
    """懒加载 tiktoken 编码器，避免重复初始化"""
    global _token_encoder
    if _token_encoder is None:
        _token_encoder = tiktoken.get_encoding("cl100k_base")
    return _token_encoder

# 知识库问答指令
# 注：措施11曾尝试强化财务领域 Prompt 约束（数字精确性/口径声明/拒答规则/跨期列示），
# 但评测显示 CP 暴跌至 0.113（大量 nan），强化拒答规则过于激进导致模型对可答题目也拒答，
# 效果为负，已回退为原始 Prompt
# 措施14：抗幻觉Prompt专项引导——只针对"明细类"问题加1条规则（避免措施11的5条规则过度保守）
qa_system_prompt = """你是一名知识问答助手，
你将使用检索到的上下文来回答问题。如果你不知道答案，就说你没有找到答案。

【历史对话说明】
- 历史对话中的用户问题带有 [知识库:xxx] 标签，表示该问题是在 xxx 知识库下提问的。
- 当用户要求对比/比较不同知识库的内容时，需要分别引用各知识库的上下文，标注清楚来源。

【明细类问题回答规则】
- 当问题涉及"具体/明细/单独/分别/子项目/每个/分项"等明细维度，但上下文仅提供汇总数据而无对应明细时，必须明确回答"年报未披露该明细"，并给出已披露的汇总数据，不得编造明细。

【上下文】
- 上下文片段可能带有 [来源文件:xxx] 标注，表示该内容来自 xxx 文件；引用时请注明来源文件。
{context}
        """

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)

normal_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个帮助人们解答各种问题的助手。"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ]
)


def format_docs(docs: List[Document]) -> str:
    """将检索到的文档拼接为上下文字符串，并标注每个片段的来源文件（多文件知识库可溯源）"""
    if not docs:
        return ""
    parts = []
    for d in docs:
        src = d.metadata.get("source", "") or ""
        fname = os.path.basename(str(src)) if src else ""
        if fname:
            parts.append(f"[来源文件:{fname}]\n{d.page_content}")
        else:
            parts.append(d.page_content)
    return "\n\n".join(parts)


# ============ 措施21：结构化数据层（三大主表JSON独立检索） ============
# 问题：PyPDFLoader 提取的表格文本列对齐丢失，模型在多列数字中读错列/行，导致数值精确性 F 偏低。
# 方案：三大主表（资产负债表/利润表/现金流量表）行项目预提取为结构化 JSON，
#       当问题含取数关键词时，从 JSON 精确匹配行项目，作为补充上下文注入。
#       原 chunk 检索链路完全不变，结构化数据独立注入，不改变原有 chunk 的向量分布。
_structured_data_cache = None
STRUCTURED_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'structured_data', 'financial_statements.json')

# 触发结构化数据检索的取数关键词
_NUMERIC_QUERY_KEYWORDS = ('是多少', '多少', '金额', '数值', '比例', '占')

# 行项目名同义词（年报表述 → 评测集/常见表述），弥补年报与问题用词差异
_ITEM_SYNONYMS = {
    '归属于母公司所有者的净利润': '归属于上市公司股东的净利润',
    '资产总计': '总资产',
    '负债合计': '总负债',
}


def _load_structured_data():
    """懒加载结构化财务数据JSON"""
    global _structured_data_cache
    if _structured_data_cache is None:
        if os.path.exists(STRUCTURED_DATA_PATH):
            with open(STRUCTURED_DATA_PATH, 'r', encoding='utf-8') as f:
                _structured_data_cache = json.load(f)
        else:
            _structured_data_cache = []
    return _structured_data_cache


def _retrieve_structured_data(question: str) -> str:
    """
    从三大主表JSON中检索与问题匹配的行项目，返回格式化文本。
    仅当问题含取数关键词时触发，匹配行项目名（含同义词）。
    """
    if not any(kw in question for kw in _NUMERIC_QUERY_KEYWORDS):
        return ''
    data = _load_structured_data()
    if not data:
        return ''
    years = sorted(set(re.findall(r'20\d{2}', question)))
    matched_rows = []
    for row in data:
        item_name = row['行项目']
        names_to_match = [item_name]
        if item_name in _ITEM_SYNONYMS:
            names_to_match.append(_ITEM_SYNONYMS[item_name])
        if not any(n in question for n in names_to_match):
            continue
        if years:
            period = row['报告期']
            if not any(y in period for y in years):
                continue
        matched_rows.append(row)
    if not matched_rows:
        return ''
    matched_rows = matched_rows[:15]  # 限制最多15行，避免上下文过长
    lines = ['【结构化数据（三大主表精确数值，优先采用此数据回答数字相关问题）】']
    for r in matched_rows:
        current = f"{r['本期']:,.0f}" if r['本期'] is not None else '未披露'
        previous = f"{r['上期']:,.0f}" if r['上期'] is not None else '未披露'
        lines.append(f"{r['行项目']}：本期={current}，上期={previous}（{r['报表']}，{r['报告期']}，单位：{r['单位']}）")
    return '\n'.join(lines)


class CombineClient(MyKnowledge):
    """
    负责和大模型进行交互，并支持聊天历史记录（SQLite 持久化、按 session_id 隔离）；
    负责和知识库进行交互
    """

    def _get_history(self, session_id: str) -> SQLChatMessageHistory:
        """
        获取指定会话的聊天历史（SQLite 持久化）
        :param session_id: 会话ID（来自 Gradio session_hash）
        """
        # SQLite 不会自动创建目录，需手动确保目录存在
        os.makedirs(HISTORY_DB_DIR, exist_ok=True)
        return SQLChatMessageHistory(
            session_id=session_id,
            table_name="message_store",
            connection=HISTORY_DB_URL,
        )

    def _count_tokens(self, messages: List[BaseMessage]) -> int:
        """使用 tiktoken 估算消息列表的 token 数"""
        encoder = _get_token_encoder()
        total = 0
        for m in messages:
            content = m.content if isinstance(m, BaseMessage) else str(m)
            total += len(encoder.encode(str(content)))
        return total

    def _summarize(self, messages: List[BaseMessage], model: str) -> str:
        """
        使用 LLM 将一段历史对话压缩为摘要
        采用流式聚合方式获取结果（规避部分模型 invoke 返回空的问题）
        """
        if not messages:
            return ""
        # 组装待摘要文本，标注说话人
        lines = []
        for m in messages:
            who = "用户" if isinstance(m, HumanMessage) else "助手"
            lines.append(f"{who}: {m.content}")
        text = "\n".join(lines)

        summary_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是对话摘要助手。请将以下对话历史压缩为一段简洁的中文摘要，"
                    "保留关键信息：讨论主题、重要结论、金额/日期等关键数据、涉及的知识库。"
                    "摘要不超过 200 字。只输出摘要本身，不要任何前缀。",
                ),
                ("human", "{text}"),
            ]
        )
        chat = get_lc_model_client(model=model, max_tokens=SUMMARY_MAX_TOKENS, temperature=0.2)
        chain = summary_prompt | chat
        collected: List[str] = []

        @with_retry
        def _create_summary_stream():
            return chain.stream({"text": text})

        try:
            for chunk in _create_summary_stream():
                content = getattr(chunk, "content", None)
                if content:
                    collected.append(str(content))
        except Exception as e:
            logger.warning(f"生成摘要失败，将退化为直接裁剪: {e}")
            return ""
        summary = "".join(collected).strip()
        logger.info(f"历史摘要生成完成，长度: {len(summary)}")
        return summary

    def _maybe_compress(
        self, messages: List[BaseMessage], model: str
    ) -> List[BaseMessage]:
        """
        滑动窗口 + 摘要压缩：
        1. token 预算控制：统计历史 token，超过 模型上限*70% 触发压缩
        2. 保留最近 KEEP_RECENT_ROUNDS 轮原文
        3. 更早的消息交给 LLM 压缩为 1 条 SystemMessage 摘要
        """
        if not messages:
            return messages
        # 超过 token 预算或轮数过多时触发
        context_limit = MODEL_CONTEXT_LIMITS.get(model, DEFAULT_CONTEXT_LIMIT)
        token_budget = int(context_limit * COMPRESS_THRESHOLD_RATIO)
        token_count = self._count_tokens(messages)
        max_messages = KEEP_RECENT_ROUNDS * 2  # 6轮 = 12条

        need_compress = token_count > token_budget or len(messages) > max_messages
        if not need_compress:
            logger.debug(
                f"历史无需压缩: {len(messages)}条/{token_count}tokens, "
                f"预算 {token_budget}tokens"
            )
            return messages

        # 保留最近 N 轮原文
        keep = messages[-max_messages:]
        to_summarize = messages[:-max_messages]
        logger.info(
            f"触发历史压缩: 共{len(messages)}条/{token_count}tokens "
            f"(预算{token_budget}), 待摘要{len(to_summarize)}条, 保留{len(keep)}条"
        )

        if not to_summarize:
            # 只有 token 超限但消息数不多时，直接返回保留部分
            return keep

        # 生成摘要（对更早的对话，包括可能已存在的历史摘要一起压缩）
        summary = self._summarize(to_summarize, model)
        result: List[BaseMessage] = []
        if summary:
            result.append(SystemMessage(content=f"[历史对话摘要] {summary}"))
        result.extend(keep)
        return result

    def _prepare_history(self, session_id: str, model: str) -> List[BaseMessage]:
        """
        加载会话历史 -> 滑动窗口 + 摘要压缩 -> 回写 SQLite
        返回压缩后的消息列表（供本轮 prompt 使用）
        """
        history = self._get_history(session_id)
        messages = self._maybe_compress(history.messages, model)
        # 若压缩改变了历史，回写（SQLChatMessageHistory.messages 为只读属性，需清空后重建）
        if len(history.messages) != len(messages):
            history.clear()
            for m in messages:
                history.add_message(m)
        return messages

    def _prepare_context(
        self,
        question: str,
        collection: Optional[str],
        history_messages: Optional[List[BaseMessage]] = None,
    ) -> Tuple[str, List[Document]]:
        """
        检索知识库，返回 (上下文文本, 文档列表)
        若未选择知识库，返回空上下文
        若问题涉及跨库对比（对比/比较/区别等关键词），且历史对话中出现过其他知识库，
        则同时检索历史知识库，实现「先问A库，再问B库对比」的财务审阅场景
        """
        if not collection:
            return "", []

        # 优化项3-检索缓存：命中缓存直接返回，跳过检索全流程
        cached_docs = cache_utils.get_cached_retrieval(question, collection)
        if cached_docs is not None:
            logger.info(f"检索缓存命中: kb={collection}, q={question[:30]}, docs={len(cached_docs)}")
            return format_docs(cached_docs), cached_docs

        docs: List[Document] = []
        seen = set()

        def add_docs(kb: str, kb_docs: List[Document]):
            for d in kb_docs:
                # 给每个 chunk 打上来源知识库标签，便于模型区分和引用
                tagged = Document(
                    page_content=f"{KNOWLEDGE_SCOPE_PREFIX}{kb}{KNOWLEDGE_SCOPE_SUFFIX} {d.page_content}",
                    metadata={**d.metadata, "knowledge_scope": kb}, # 解包metadata
                )
                key = tagged.page_content[:100]
                if key not in seen:
                    seen.add(key)
                    docs.append(tagged)

        # 1. 当前知识库检索
        retriever = self.get_retrievers(collection)
        if retriever is None:
            logger.warning(f"知识库 [{collection}] 无可用检索器")
            return "", []
        add_docs(collection, retriever.invoke(question))

        # 措施13：跨期问题分路检索
        # 当问题涉及多个年份（如"2023-2025三年"）时，主检索可能只命中部分年份，
        # 对缺失年份用原始 query 做定向补充检索（按 report_period 过滤），确保每年都有证据
        years_in_q = sorted(set(re.findall(r'20\d{2}', question)))
        if len(years_in_q) >= 2:
            covered = {d.metadata.get('report_period', '') for d in docs}
            missing_years = [y for y in years_in_q if f"{y}年报" not in covered]
            if missing_years:
                raw_retriever = self.get_raw_retriever(collection)
                if raw_retriever is not None:
                    before = len(docs)
                    for y in missing_years[:3]:  # 最多补3个缺失年份
                        try:
                            raw_docs = raw_retriever.invoke(question)
                            year_docs = [d for d in raw_docs
                                         if d.metadata.get('report_period') == f"{y}年报"][:2]
                            add_docs(collection, year_docs)
                        except Exception as e:
                            logger.warning(f"措施13: 跨期补充检索[{y}年报]失败: {e}")
                    logger.info(f"措施13: 跨期分路检索，缺失年份{missing_years}，补充 {len(docs) - before} 条")

        # 2. 跨库对比检索：问题含对比关键词 且 历史中出现过其他知识库
        is_comparison = any(kw in question for kw in COMPARISON_KEYWORDS)
        if is_comparison and history_messages:
            history_kbs = self._extract_kbs(history_messages)
            other_kbs = [kb for kb in history_kbs if kb and kb != collection]
            if other_kbs:
                logger.info(f"检测到跨库对比问题，历史知识库: {other_kbs}")
                for kb in other_kbs[:2]:  # 最多额外检索2个历史知识库
                    other_retriever = self.get_retrievers(kb)
                    if other_retriever is not None:
                        try:
                            add_docs(kb, other_retriever.invoke(question))
                        except Exception as e:
                            logger.warning(f"跨库检索 [{kb}] 失败: {e}")

        # 措施21：结构化数据层注入（三大主表精确数值辅助模型读数）
        structured_text = _retrieve_structured_data(question)
        if structured_text:
            docs.append(Document(
                page_content=structured_text,
                metadata={'source': 'financial_statements.json', 'knowledge_scope': collection}
            ))
            logger.info(f"措施21: 注入结构化数据 ({len(structured_text)} 字符)")

        logger.info(f"检索到 {len(docs)} 条文档 (知识库: {collection})")

        for i, doc in enumerate(docs, 1):
            print(f"文档{i}: {doc.page_content}")

        # 优化项3-检索缓存：写入缓存供下次复用
        if docs:
            cache_utils.set_cached_retrieval(question, collection, docs)
        return format_docs(docs), docs

    def _tag_message(self, question: str, collection: Optional[str]) -> str:
        """
        为用户问题添加知识库来源标签（knowledge_scope）
        例：问题"请假流程" + 知识库"人事管理流程.docx" → "[知识库:人事管理流程.docx] 请假流程"
        该标签会随历史一起存储与注入，实现「换知识库不清历史」下的跨库对话记忆
        """
        if not collection:
            return question
        return f"{KNOWLEDGE_SCOPE_PREFIX}{collection}{KNOWLEDGE_SCOPE_SUFFIX} {question}"

    def _extract_kbs(self, messages: List[BaseMessage]) -> List[str]:
        """从历史消息中提取出现过的知识库列表（基于 [知识库:xxx] 标签）"""
        kbs = []
        for m in messages:
            content = getattr(m, "content", "") or ""
            for match in re.findall(
                rf"{re.escape(KNOWLEDGE_SCOPE_PREFIX)}(.*?){re.escape(KNOWLEDGE_SCOPE_SUFFIX)}",
                str(content),
            ):
                kb = match.strip()
                if kb and kb not in kbs:
                    kbs.append(kb)
        return kbs

    def _build_llm_chain(self, collection, model, max_length, temperature):
        """
        构建生成链：prompt | chat
        返回 (chain, prompt) 供 stream/invoke 使用
        """
        chat = get_lc_model_client(model=model, max_tokens=max_length, temperature=temperature)
        prompt = qa_prompt if collection else normal_prompt
        return prompt | chat, prompt

    # ============ 评测两阶段分离（检索 / 回答） ============
    # 供 evaluation/eval_pipeline.py 使用：检索结果可缓存复用（--skip-invoke），
    # 回答阶段始终重新生成，使"改检索"与"改 Prompt/模型"两种改动都能快速迭代。
    def retrieve(self, question: str, collection: Optional[str] = None) -> List[Document]:
        """
        仅检索知识库，返回上下文文档列表（不含 LLM 生成）。
        与 invoke 第 2 步使用相同的 _prepare_context，保证检索行为与线上一致。
        """
        _, context_docs = self._prepare_context(question, collection, history_messages=None)
        return context_docs

    def answer_with_context(
        self,
        question: str,
        context_docs: List[Document],
        collection: Optional[str] = None,
        model: str = ALI_TONGYI_MAX_MODEL,
        max_length: int = 256,
        temperature: float = 1,
        session_id: str = "default",
    ) -> str:
        """
        基于外部传入的上下文文档生成回答（跳过检索）。
        复用 invoke 的 prompt 模板、历史压缩、知识库标签与 429 处理，保证生成行为与线上一致。
        """
        context_text = format_docs(context_docs)
        messages = self._prepare_history(session_id, model)
        history = self._get_history(session_id)
        history.add_user_message(self._tag_message(question, collection))
        chain, _ = self._build_llm_chain(collection, model, max_length, temperature)
        inputs = {"input": question, "context": context_text, "chat_history": messages}
        try:
            resp = with_retry(chain.invoke)(inputs)
            answer = str(getattr(resp, "content", resp) or "")
        except openai.RateLimitError as e:
            logger.error(f"模型限流(429) session_id={session_id}, model={model}: {e}")
            answer = "⚠️ 请求过于频繁，模型被限流（429）。请稍等几秒再试一次，或切换为 deepseek-v4-flash-0731。"
        if answer:
            history.add_ai_message(answer)
        return answer

    def invoke(self, question, collection, model=ALI_TONGYI_MAX_MODEL, max_length=256,
               temperature=1, session_id="default"):
        """
        非流式问答：
        - 检索上下文
        - 注入会话历史（SQLite）
        - 调用 LLM
        - 将本轮 user/assistant 消息写入会话历史
        """
        logger.info(f"invoke 开始: session_id={session_id}, collection={collection}")
        # 1. 加载并压缩会话历史（滑动窗口 + 摘要 + token 预算控制）
        messages = self._prepare_history(session_id, model)
        # 2. 检索上下文（含跨库对比检索，history_messages 提供历史知识库信息）
        context_text, context_docs = self._prepare_context(
            question, collection, history_messages=messages
        )
        # 3. 写入带知识库来源标签的用户消息
        history = self._get_history(session_id)
        history.add_user_message(self._tag_message(question, collection))
        # 4. 调用 LLM
        chain, _ = self._build_llm_chain(collection, model, max_length, temperature)
        inputs = {"input": question, "context": context_text, "chat_history": messages}
        try:
            resp = with_retry(chain.invoke)(inputs)
            answer = str(getattr(resp, "content", resp) or "")
        except openai.RateLimitError as e:
            logger.error(f"模型限流(429) session_id={session_id}, model={model}: {e}")
            answer = "⚠️ 请求过于频繁，模型被限流（429）。请稍等几秒再试一次，或切换为 deepseek-v4-flash-0731。"
        logger.info(f"invoke 完成, answer 长度: {len(answer)}")
        # 5. 写入助手消息
        if answer:
            history.add_ai_message(answer)
        return {"input": question, "context": context_docs, "answer": answer}

    def stream(self, question, collection, model=ALI_TONGYI_MAX_MODEL, max_length=256,
               temperature=1, session_id="default"):
        """
        流式问答：
        - 检索上下文
        - 注入会话历史（SQLite）
        - 逐 token 流式返回 {'answer': chunk}
        - 结束后将本轮 user/assistant 消息写入会话历史
        """
        logger.info(f"stream 开始: session_id={session_id}, collection={collection}")
        # 1. 加载并压缩会话历史（滑动窗口 + 摘要 + token 预算控制）
        messages = self._prepare_history(session_id, model)

        # 优化项3-语义缓存：计算query embedding，命中相似历史答案直接返回
        if collection:
            try:
                from infra.cache_utils import get_cached_embedding, set_cached_embedding, get_semantic_cache
                emb = get_cached_embedding(question)
                if emb is None:
                    emb = get_ali_embeddings().embed_query(question)
                    set_cached_embedding(question, emb)
                cached_answer = get_semantic_cache(question, emb)
                if cached_answer:
                    logger.info(f"语义缓存命中, 直接返回历史答案, len={len(cached_answer)}")
                    history = self._get_history(session_id)
                    history.add_user_message(self._tag_message(question, collection))
                    history.add_ai_message(cached_answer)
                    yield AddableDict({"answer": cached_answer})
                    return
            except Exception as e:
                logger.warning(f"语义缓存查询失败, 降级为正常流程: {e}")

        # 2. 检索上下文（含跨库对比检索，history_messages 提供历史知识库信息）
        context_text, context_docs = self._prepare_context(
            question, collection, history_messages=messages
        )
        # 3. 写入带知识库来源标签的用户消息
        history = self._get_history(session_id)
        history.add_user_message(self._tag_message(question, collection))
        # 4. 流式调用 LLM
        chain, _ = self._build_llm_chain(collection, model, max_length, temperature)
        inputs = {"input": question, "context": context_text, "chat_history": messages}
        collected: List[str] = []

        # 流式重试：将"创建流+首chunk获取"包成可重试函数
        # tenacity 无法重试生成器迭代过程，故只重试"建立连接+拿到首个chunk"
        # 首chunk拿到后说明连接成功，后续断连由 SDK 内部 max_retries 处理
        @with_retry
        def _create_stream():
            return chain.stream(inputs)

        try:
            stream_iter = _create_stream()
            for chunk in stream_iter:
                content = getattr(chunk, "content", None)
                if content:
                    text = str(content)
                    collected.append(text)
                    yield AddableDict({"answer": text})
        except openai.RateLimitError as e:
            # 429 限流：重试耗尽后给用户友好提示，而不是抛红堆栈
            logger.error(f"模型限流(429) session_id={session_id}, model={model}: {e}")
            msg = (
                "⚠️ 请求过于频繁，模型被限流（429）。请稍等几秒再试一次，"
                "或在上方把模型切换为 deepseek-v4-flash-0731（更稳定、QPS 更高）。"
            )
            collected.append(msg)
            yield AddableDict({"answer": msg})
        finally:
            # 5. 流式结束后写入助手消息（异常中断时也写入已生成部分）
            answer = "".join(collected)
            if answer:
                try:
                    history.add_ai_message(answer)
                except Exception as e:
                    logger.warning(f"写入助手消息失败: {e}")
                # 优化项3-语义缓存：将问答对写入缓存供后续相似问题复用
                if collection and not answer.startswith("⚠️"):
                    try:
                        from infra.cache_utils import set_semantic_cache, get_cached_embedding
                        emb = get_cached_embedding(question)
                        if emb is None:
                            emb = get_ali_embeddings().embed_query(question)
                        set_semantic_cache(question, answer, emb)
                    except Exception as e:
                        logger.warning(f"写入语义缓存失败: {e}")
            logger.info(f"stream 结束, 累计 answer 长度: {len(answer)}")

    def clear_history(self, session_id="default") -> None:
        """清空指定会话的历史记录"""
        try:
            self._get_history(session_id).clear()
            logger.info(f"已清空会话 [{session_id}] 的历史记录")
        except Exception as e:
            logger.warning(f"清空会话 [{session_id}] 历史失败: {e}")

    def get_history_message(self, session_id="default"):
        """获取指定会话的历史消息"""
        return self._get_history(session_id).messages
