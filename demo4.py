import asyncio
import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv(override=True)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="你是乐于助人的助手，请根据用户的问题给出回答"
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

model = ChatTongyi(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model = "qwen-turbo"
)

chain = prompt | model | StrOutputParser()

messages_list = []

# 创建异步函数处理流式响应
async def get_streaming_response(chain, messages_list):
    response = ""
    print("🤖 猫娘：", end="", flush=True)
    async for chunk in chain.astream({"messages": messages_list}):
        response += chunk
        print(chunk, end="", flush=True)
    print()  # 换行
    return response

while True:
    user_input = input("👤 你：")
    if user_input in ["quit", "q", "exit"]:
        break

    messages_list.append(HumanMessage(content=user_input))
    # response = chain.invoke({
    #     "messages":messages_list
    # })
    # print(f"🤖 猫娘：{response}")

    # 流式回答
    response = asyncio.run(get_streaming_response(chain, messages_list))

    messages_list.append(AIMessage(content=response))
    messages_list = messages_list[-50:]


# # 模拟对话
# messages_list = [
#     HumanMessage(content="你好，我是白晨，请你介绍一下你自己"),
#     AIMessage(content="你好，我是猫娘，有什么可以帮您的吗？"),
# ]
#
# question = "你现在感觉怎么样？"
#
# messages_list.append(HumanMessage(content=question))
#
# print(messages_list)

# response = chain.invoke({
#     "messages":messages_list
# })
# print(response)