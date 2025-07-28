import os

import requests
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv(override=True)


class WeatherQuery(BaseModel):
    loc: str = Field(description="城市名称，比如 'Beijing', 'Shanghai' 等")


@tool(args_schema=WeatherQuery)
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


search_tool = TavilySearch(tavily_api_key=os.getenv("TAVILY_API_KEY"))

tools = [get_weather, search_tool]

model = ChatTongyi(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

checkpointer = InMemorySaver()

agent = create_react_agent(model=model, tools=tools, checkpointer=checkpointer)

config = {
    "configurable": {
        "thread_id": "1",
        # "recursion_limit":  4
    },
}

response = agent.invoke({"messages":
                             [{"role": "user", "content": "你好，我是白晨"}]
                         },
                         config
                        )

print(len(response["messages"]))
print(response["messages"][-1].content)


response = agent.invoke({"messages":
                             [{"role": "user", "content": "你好，你记得我是谁吗？"}]
                         },
                         config
                        )
print(response["messages"][-1].content)

lastest = agent.get_state(config)
print(lastest)