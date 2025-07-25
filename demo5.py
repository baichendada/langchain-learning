import os

import gradio as gr
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv(override=True)

# ──────────────────────────────────────────────
# 1. 模型、Prompt、Chain
# ──────────────────────────────────────────────
model = ChatTongyi(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
parser = StrOutputParser()

chatbot_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="你叫小智，是一名乐于助人的助手。"),
        MessagesPlaceholder(variable_name="messages"),  # 手动传入历史
    ]
)

qa_chain = chatbot_prompt | model | parser   # LCEL 组合

# ──────────────────────────────────────────────
# 2. Gradio 组件
# ──────────────────────────────────────────────
CSS = """
.main-container {max-width: 1200px; margin: 0 auto; padding: 20px;}
.header-text {text-align: center; margin-bottom: 20px;}
"""

def create_chatbot() -> gr.Blocks:
    with gr.Blocks(title="Qwen Chat", css=CSS) as demo:
        with gr.Column(elem_classes=["main-container"]):
            gr.Markdown("# 🤖 LangChain", elem_classes=["header-text"])
            gr.Markdown("基于 LangChain 构建的流式对话机器人", elem_classes=["header-text"])

            chatbot = gr.Chatbot(
                height=500,
                show_copy_button=True,
                type='messages',
                avatar_images=(
                    "https://cdn.jsdelivr.net/gh/twitter/twemoji@v14.0.2/assets/72x72/1f464.png",
                    "https://cdn.jsdelivr.net/gh/twitter/twemoji@v14.0.2/assets/72x72/1f916.png",
                ),
            )
            msg = gr.Textbox(placeholder="请输入您的问题...", container=False, scale=7)
            submit = gr.Button("发送", scale=1, variant="primary")
            clear = gr.Button("清空", scale=1)

        # ---------------  状态：保存 messages_list  ---------------
        state = gr.State([])          # 这里存放真正的 Message 对象列表

        # ---------------  主响应函数（流式） ----------------------
        async def respond(user_msg: str, chat_hist: list, messages_list: list):
            # 1) 输入为空直接返回
            if not user_msg.strip():
                yield "", chat_hist, messages_list
                return

            # 2) 追加用户消息
            messages_list.append(HumanMessage(content=user_msg))
            chat_hist = chat_hist + [{"role": "user", "content": user_msg}]
            yield "", chat_hist, messages_list      # 先显示用户消息

            # 3) 流式调用模型
            partial = ""
            async for chunk in qa_chain.astream({"messages": messages_list}):
                partial += chunk
                # 更新最后一条 AI 回复
                if len(chat_hist) > 0 and chat_hist[-1]["role"] == "user":
                    chat_hist.append({"role": "assistant", "content": partial})
                else:
                    chat_hist[-1] = {"role": "assistant", "content": partial}
                yield "", chat_hist, messages_list

            # 4) 完整回复加入历史，裁剪到最近 50 条
            messages_list.append(AIMessage(content=partial))
            messages_list = messages_list[-50:]

            # 5) 最终返回（Gradio 需要把新的 state 传回）
            yield "", chat_hist, messages_list

        # ---------------  清空函数 -------------------------------
        def clear_history():
            return [], "", []          # 清空 Chatbot、输入框、messages_list

        # ---------------  事件绑定 ------------------------------
        msg.submit(respond, [msg, chatbot, state], [msg, chatbot, state])
        submit.click(respond, [msg, chatbot, state], [msg, chatbot, state])
        clear.click(clear_history, outputs=[chatbot, msg, state])

    return demo


# ──────────────────────────────────────────────
# 3. 启动应用
# ──────────────────────────────────────────────
demo = create_chatbot()
demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=True)