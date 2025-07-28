import os
import asyncio
from typing import Annotated

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(override=True)

# 定义状态类（会自动合并 messages）
class State(TypedDict):
    messages: Annotated[list, add_messages]

model = ChatTongyi(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-turbo"
)

def chatbot(state: State) -> State:
    reply = model.invoke(state["messages"])
    return  {"messages": [reply]}

builder = StateGraph(State)

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

thread_config = {"configurable": {"thread_id": "session_10"}}

async def main():
    # state1 = graph.invoke({"messages": [{"role":"user","content":"你好，好久不见，我叫陈明。"}]}, config=thread_config)
    # print(state1["messages"][-1].content)
    # chunks = []
    # async for chunk in graph.astream({"messages": [{"role":"user","content":"你好，好久不见，我叫陈明。"}]}, config=thread_config):
    #     chunks.append(chunk)
    #     print(chunks, end="", flush=True)
    #
    # # state2 = graph.invoke({"messages": [{"role":"user","content":"你好，好久不见，你知道我叫什么吗？"}]}, config=thread_config)
    # # print(state2["messages"][-1].content)
    # async for chunk in graph.astream({"messages": [{"role":"user","content":"你好，好久不见，你知道我叫什么吗？"}]}, config=thread_config):
    #     chunks.append(chunk)
    #     print(chunks, end="", flush=True)
    async for msg, metadata in graph.astream({"messages": ["你好，请你详细的介绍一下你自己"]}, stream_mode="messages", config=thread_config):
        if msg.content and not isinstance(msg, HumanMessage):
            print(msg.content, end="", flush=True)

    latest = graph.get_state(thread_config)
    print(latest.values["messages"])

if __name__ == "__main__":
    asyncio.run(main())