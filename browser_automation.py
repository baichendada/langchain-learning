import os

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain import hub

load_dotenv(override=True)

model = ChatTongyi(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-turbo"
)

sync_browser = create_sync_playwright_browser()
toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
tools = toolkit.get_tools()

# 使用ReAct prompt，兼容性更好
prompt = hub.pull("hwchase17/react")

# 使用ReAct agent，兼容ChatTongyi
agent = create_react_agent(llm=model, prompt=prompt, tools=tools)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if __name__ == "__main__":
    command = {
        "input": "访问这个网站，https://blog.csdn.net/baichendada/article/details/148673714?spm=1001.2014.3001.5501，并用中文总结这篇文章的内容"
    }

    response = agent_executor.invoke(command)
    print("\n=== 执行结果 ===")
    print(response)