import os

import requests
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import JsonOutputKeyToolsParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

load_dotenv(override=True)

# OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
#
# print(OPENWEATHER_API_KEY)

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

# print(get_weather("Beijing"))

model = ChatTongyi(
    model = "qwen-turbo",
    api_key = os.getenv("DASHSCOPE_API_KEY"),
)

llm_with_tools = model.bind_tools([get_weather])

# print(llm_with_tools.invoke("Beijing的天气如何？"))

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个天气助手"),
        ("human", "{input}")
    ]
)

parser = JsonOutputKeyToolsParser(key_name=get_weather.name, first_tool_only=True)

# print(get_weather.name)

def debug_print(x):
    print(x)
    return x

tool_chain = prompt | llm_with_tools | RunnableLambda(debug_print) | parser | RunnableLambda(debug_print) | get_weather
# print(tool_chain.invoke({"input": "北京今天的天气如何？（如果要调用外部工具查询天气，请将城市名转换为英文再查询）"}))
weather_json = tool_chain.invoke({"input": "北京今天的天气如何？（如果要调用外部工具查询天气，请将城市名转换为英文再查询）"})
print()

out_prompt = PromptTemplate.from_template(
"""你将收到一段 JSON 格式的天气数据，请用简洁自然的方式将其转述给用户。
以下是天气 JSON 数据：

```json
{weather_json}
```

请将其转换为中文天气描述，例如：
“北京当前天气晴，气温为 23°C，湿度 58%，风速 2.1 米/秒。”
只返回一句话描述，不要其他说明或解释。"""
)

output_chain = out_prompt | model | StrOutputParser()
response = output_chain.invoke({"weather_json": weather_json})
print(response)