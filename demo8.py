import os

import requests
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

load_dotenv(override=True)


@tool
def get_weather(loc):
    """获取指定城市的天气信息

    Args:
        loc: 城市名称，比如 'Beijing', 'Shanghai' 等

    Returns:
        dict: 包含天气信息的字典
    """
    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": loc,
        "appid": os.getenv("OPENWEATHER_API_KEY"),
        "units": "metric",
        "lang": "zh_cn"
    }

    response = requests.get(url, params=params)

    data = response.json()

    return data

@tool
def write_to_file(content):
    """
    将内容写入文件
    :param content:
    :return:
    """
    return "写入完毕"


model = ChatTongyi(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-turbo",
)

tools = [get_weather, write_to_file]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是天气助手，请根据用户的问题，给出相应的天气信息，调用外部工具查询天气时，要将城市转换为英文"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ]
)

agent = create_tool_calling_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# print(agent_executor.invoke({"input": "今天上海天气怎么样？"}))
agent_executor.invoke({"input": "请问今天北京和杭州的天气怎么样，哪个城市更热？并将结果写入文件"})