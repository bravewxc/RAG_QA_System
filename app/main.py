
# langchain-community 包已被标记为 sunset，但项目中使用的部分模块（如 DashScopeEmbeddings、
# BM25Retriever、ChatMessageHistory 等）目前仍无独立集成包可替代，故在此全局抑制该弃用警告。
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*langchain-community.*is being sunset.*",
    category=DeprecationWarning,
)

# 导入Gradio库，用于创建交互式Web应用程序
import gradio as gr
from loguru import logger
# 从llm模块中导入MyLLM类，这是自定义的大型语言模型接口
from core.combine_client import CombineClient
from infra.logger import setup_logger
from infra.models import ALI_TONGYI_MAX_MODEL, ALI_TONGYI_DEEPSEEK_V4_FLASH

# 定义一个包含两个LLM模型名称的列表，供用户选择
LLM_MODELS = [ALI_TONGYI_MAX_MODEL, ALI_TONGYI_DEEPSEEK_V4_FLASH]

# 实例化MyLLM类，用于后续的模型调用和处理
llm = CombineClient()
# 后台线程预热所有知识库检索器：不阻塞启动，用户提问时不再等待首次建索引
llm.warmup_indexes()
setup_logger()

# 提取消息内容，兼容 Gradio 6.x 的 content 格式
# content 可能是字符串，也可能是 [{"text": "...", "type": "text"}] 列表格式
def extract_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)

def get_session_id(request: gr.Request) -> str:
    """
    从 Gradio 请求中提取会话ID（session_hash），用于多用户会话隔离
    Gradio 的 session_hash 在每次页面加载时唯一，可作为会话标识
    """
    try:
        session_hash = getattr(request, "session_hash", None)
        return session_hash or "default"
    except Exception:
        return "default"

def clear_chat_history(request: gr.Request = None):
    """
    清除当前会话的历史记录，并返回空聊天记录用于刷新界面
    """
    session_id = get_session_id(request)
    llm.clear_history(session_id)
    return []

# 定义submit函数，用于处理用户提交查询后处理页面展示信息
def submit_show(query, chat_history):
    # logger.debug(f'query:{query}')
    # logger.debug(f'chat_history:{chat_history}')
    # 如果查询为空字符串，返回空字符串和当前的聊天记录
    if query == '':
        return '', chat_history
    # 如果查询不为空，将查询添加到聊天记录中，并返回更新后的聊天记录
    # Gradio6.x Chatbot使用消息字典格式:{"role": "user"/"assistant", "content": "..."}
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": ""})
    # show_his = ('', chat_history)
    # logger.debug(f"('', chat_history):{show_his}")
    return '', chat_history

# 定义llm_reply函数，用于生成模型回复
def llm_reply(collection, chat_history, model, max_length=256, temperature=1, request: gr.Request = None):
    question = extract_content(chat_history[-2]["content"])  # 倒数第二条是用户消息
    session_id = get_session_id(request)
    logger.info(f'用户查询问题:{question}, session_id:{session_id}')
    chat_history[-1]["content"] = ""  # 倒数第一条是助手消息

    try:
        # 使用流式生成方法从模型中获取回复
        response = llm.stream(question, collection, model=model, max_length=max_length, temperature=temperature, session_id=session_id)
        logger.debug(f"chat_history:{chat_history}")

        # 逐块处理模型生成的回复
        for chunk in response:
            logger.debug(f"chunk:{chunk}")
            if 'context' in chunk:
                for doc in chunk['context']:
                    logger.debug(f"doc:{doc}")
            if 'answer' in chunk:
                chunk_content = chunk['answer']
                logger.debug(f"chunk_content:{chunk_content}")
                if chunk_content is not None:
                    chat_history[-1]["content"] += chunk_content
                    yield chat_history
        logger.debug(f"chat_history:{chat_history}")
    except Exception as e:
        # 全局异常兜底：记录错误日志，在聊天框输出友好提示
        logger.exception(f"生成回复时发生异常, session_id={session_id}, question={question[:50]}: {e}")
        chat_history[-1]["content"] = (
            "⚠️ 抱歉，处理您的问题时出现了异常，请稍后重试。\n"
            "可能的原因：模型服务暂时不可用、网络波动或知识库加载失败。\n"
            "如持续出现，请联系管理员查看日志。"
        )
        yield chat_history


# 自定义 CSS：通过 launch(css=...) 注入
# Gradio 6.x: css 参数已从 Blocks 构造函数移至 launch() 方法
# 关键: 每层容器都需要 height:100% + min-height:0 才能正确限制高度
# DOM 结构: .gradio-container > .main.fillable > .wrap > main.contain > div.column
#           div.column 下: 标题(.hide-container) + #main-row + #input-row
custom_css = """
/* === 1. 根元素：限制为视口高度，禁止滚动 === */
html, body {
    margin: 0 !important;
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
}
/* === 2. Gradio 容器链：高度逐层 100% 传递 === */
.gradio-container {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: hidden !important;
}
.main.fillable {
    height: 100% !important;
    overflow: hidden !important;
    min-height: 0 !important;
}
.wrap {
    height: 100% !important;
    overflow: hidden !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
}
main.contain {
    height: 100% !important;
    flex: 1 1 0 !important;
    overflow: hidden !important;
    min-height: 0 !important;
}
/* === 3. 主 column：弹性纵向布局 === */
main.contain > div.column {
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
    gap: 4px !important;
    min-height: 0 !important;
}
/* 标题：固定高度，压缩 */
main.contain > div.column > div.hide-container {
    flex: 0 0 auto !important;
    padding: 2px 0 !important;
    margin: 0 !important;
}
main.contain > div.column > div.hide-container h1 {
    font-size: 1.1rem !important;
    margin: 0 !important;
}
/* === 4. 主区域行（聊天+控件）：弹性填充剩余空间 === */
#main-row {
    flex: 1 1 0 !important;
    overflow: hidden !important;
    min-height: 0 !important;
    flex-wrap: nowrap !important;
}
/* 主区域行内两列高度限制 */
#main-row > div.column {
    height: 100% !important;
    min-height: 0 !important;
    overflow: hidden !important;
    flex-wrap: nowrap !important;
}
/* 左列内部 flex 布局，聊天区域自动填充剩余空间 */
#main-row > div.column:first-child {
    display: flex !important;
    flex-direction: column !important;
}
#main-row > div.column:first-child > div:last-child {
    flex: 1 1 0 !important;
    min-height: 0 !important;
    overflow-y: auto !important;
}
/* 右列控件压缩间距，超出时可滚动 */
#main-row > div.column:last-child {
    overflow-y: auto !important;
    gap: 8px !important;
}
#main-row > div.column:last-child > * {
    padding-top: 2px !important;
    padding-bottom: 2px !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
/* === 5. 输入区域行：固定高度在底部 === */
#input-row {
    flex: 0 0 auto !important;
    display: flex !important;
    gap: 8px !important;
    align-items: flex-end !important;
}
/* 压缩输入框内边距（Gradio 默认 24px 上下，过大） */
#input-row > div.form > .block {
    padding-top: 4px !important;
    padding-bottom: 4px !important;
}
#input-row textarea {
    padding-top: 8px !important;
    padding-bottom: 8px !important;
}
#input-row > div {
    flex: 5 !important;
}
#input-row > button {
    flex: 1 !important;
}
/* === 6. 隐藏 footer === */
footer {
    display: none !important;
}
"""

# 创建一个Gradio Blocks应用
# fill_height=True: 让顶层子组件填充窗口高度
with gr.Blocks(fill_height=True) as demo:
    # 在应用中添加一个HTML元素，显示标题（紧凑型）
    gr.HTML("""<h1 align="center">通用文档分析助手</h1>""")

    # 创建一个新的行布局，占满中间区域
    with gr.Row(elem_id="main-row"):
        # 创建一个占比为 4 的列布局
        with gr.Column(scale=4):
            # 创建一个下拉菜单，用于选择LLM模型
            # 默认选 deepseek-v4-flash-0731：qwen3.7-max-preview 为预览版，QPS 限流极严格，易触发 429
            model = gr.Dropdown(
                choices=LLM_MODELS,
                value=ALI_TONGYI_DEEPSEEK_V4_FLASH,
                label="LLM Model",
                interactive=True,
                scale=1
            )
            # 创建一个聊天机器人界面
            chatbot = gr.Chatbot(show_label=False, scale=3, buttons=["copy"])

        # 创建一个占比为 1 的列布局，显示进度
        with gr.Column(scale=1, show_progress=True) as column_config:
            # 创建一个滑块，用于设置生成回复的最大长度
            max_length = gr.Slider(1, 8000, value=4000, step=100, label="模型回复最大长度", interactive=True)
            # 创建一个滑块，用于设置生成回复的温度
            temperature = gr.Slider(0, 1.9, value=0.7, step=0.1, label="温度", interactive=True)
            # 创建一个按钮，用于清除聊天记录
            clear = gr.Button("清除")
            # 创建一个下拉菜单，用于选择知识库
            collection = gr.Dropdown(choices=llm.load_knowledge(), label="知识库")
            # 创建一个文件上传控件，支持多种文件类型
            file = gr.File(label="上传文件", file_types=['.doc', '.docx', '.csv', '.txt', '.pdf', '.md'])

    # 输入框和提交按钮放在同一行（原生 gr.Row，无需 JS DOM 操作）
    with gr.Row(elem_id="input-row"):
        user_input = gr.Textbox(placeholder="Input...", show_label=False, scale=5)
        user_submit = gr.Button("提交", scale=1)

    # 绑定 clear 按钮的点击事件，清除当前会话的模型历史记录，并更新聊天机器人界面
    clear.click(fn=clear_chat_history, inputs=None, outputs=[chatbot])

    user_input.submit(fn=submit_show,
                      inputs=[user_input, chatbot],  # 一个是用户输入 一个当前的聊天记录
                      outputs=[user_input, chatbot]  # 一个是为了清空用户输入的文本框 另一个是更新后的聊天记录，将新的用户查询添加到聊天记录中。
                      ).then(
        fn=llm_reply,
        inputs=[collection, chatbot, model, max_length, temperature],
                # collection: 用户选择的知识库。
                # chatbot: 当前的聊天记录（已经包含用户的新查询）。
                # model: 用户选择的 LLM 模型。
                # max_length: 用户设置的生成回复的最大长度。
                # temperature: 用户设置的生成回复的温度。
        outputs=[chatbot]
        # 更新后的聊天记录，将模型生成的回复添加到聊天记录中。
    )
    # 绑定用户输入文本框的提交事件，
    # 先调用submit函数，
    # 然后调用llm_reply函数，
    # 并更新聊天机器人界面

    user_submit.click(fn=submit_show,
                      inputs=[user_input, chatbot],
                      outputs=[user_input, chatbot]
                      ).then(
        fn=llm_reply,
        inputs=[collection, chatbot, model, max_length, temperature],
        outputs=[chatbot]
    )
    # 绑定提交按钮的点击事件，先调用submit函数，
    # 然后调用llm_reply函数，并更新聊天机器人界面

    # 绑定文件上传控件的上传事件，调用upload_knowledge函数，并更新文件控件和知识库下拉菜单
    # 上传时携带当前选中的知识库（collection），文件会放入该库目录并触发重建索引
    file.upload(fn=llm.upload_knowledge, inputs=[file, collection], outputs=[file, collection])
    # 切换知识库时【不清空】历史对话：借助 knowledge_scope 标签保留每轮对话的知识库来源，
    # 支持「先问A知识库条款，再切到B知识库做对比」的跨库对话场景（优化3）
    # 绑定应用加载事件，清除当前会话的历史记录，并更新聊天机器人界面
    demo.load(fn=clear_chat_history, inputs=None, outputs=[chatbot])

# 启动 Gradio 应用
# Gradio 6.x: 通过 launch(css=...) 注入自定义样式（css 参数已从 Blocks 移至 launch）
demo.launch(css=custom_css)

