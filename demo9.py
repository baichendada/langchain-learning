import os

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch
from langchain.agents import create_tool_calling_agent, AgentExecutor

load_dotenv(override=True)

search = TavilySearch(tavily_api_key=os.getenv("TAVILY_API_KEY"))

# print(search.invoke("苹果2025WWDC发布会"))

tools = [search]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一名助人为乐的助手，并且可以调用工具进行网络搜索，获取实时信息。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# 添加具体模型参数
model = ChatTongyi(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-turbo"
)

# 直接传入原始模型到agent
agent = create_tool_calling_agent(llm=model, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 执行并打印结果
result = agent_executor.invoke({"input": "苹果2025WWDC发布会召开的时间是什么时候？"})
print("\n=== 执行结果 ===")
print(result)