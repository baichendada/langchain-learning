import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.chat_models import init_chat_model
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv(override=True)
print(os.getenv("LLM_API_KEY"))
print(os.getenv("MODEL"))

class Configuration:
    """读取 .env 与 servers_config.json"""
    def __init__(self) -> None:
        
        self.api_key: str = os.getenv("LLM_API_KEY") or ""
        self.model: str = os.getenv("MODEL") or "deepseek-chat"
        if not self.api_key:
            raise ValueError("❌ 未找到 LLM_API_KEY，请在 .env 中配置")

    @staticmethod
    def load_servers(file_path: str = "servers_config.json") -> Dict[str, Any]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f).get("mcpServers", {})

async def get_mcp_tools(mcp_client):
    """手动获取MCP工具，避免使用有问题的load_mcp_tools函数"""
    try:
        # 尝试使用get_tools方法
        if hasattr(mcp_client, 'get_tools'):
            tools = await mcp_client.get_tools()
            return tools
        
        # 如果没有get_tools，尝试其他方法
        elif hasattr(mcp_client, 'list_tools'):
            tools = await mcp_client.list_tools()
            return tools
        
        # 如果都没有，返回空列表
        else:
            print("⚠️ MCP客户端没有可用的工具获取方法")
            return []
            
    except Exception as e:
        print(f"⚠️ 获取MCP工具时出错: {e}")
        return []
    
async def run_chat_loop() -> None:
    """主循环"""
    cfg = Configuration()
    servers_cfg = cfg.load_servers()

    os.environ["OPENAI_API_KEY"] = cfg.api_key

    try:
        mcp_client = MultiServerMCPClient(servers_cfg)
        
        # 使用我们自己的工具获取函数
        tools = await get_mcp_tools(mcp_client)
        
        if tools:
            logging.info(f"✅ 已加载 {len(tools)} 个 MCP 工具： {[getattr(t, 'name', str(t)) for t in tools]}")
        else:
            logging.warning("⚠️ 没有加载到MCP工具，将使用基础聊天功能")
            tools = []  # 空工具列表，仍然可以进行基础对话

    except Exception as e:
        print(f"⚠️ MCP客户端初始化失败: {e}")
        print("💡 将使用基础聊天功能（无MCP工具）")
        tools = []
        mcp_client = None

    model = ChatTongyi(
        api_key=cfg.api_key,
        model=cfg.model,
    )

    # 根据是否有工具决定是否创建agent
    if tools:
        prompt = hub.pull("hwchase17/openai-tools-agent")
        
        agent = create_openai_tools_agent(
            llm=model,
            tools=tools,
            prompt=prompt,
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True
        )
        
        print("\n🤖 MCP Agent 已启动（带工具），输入 'quit' 退出")
    else:
        # 没有工具时，直接使用模型
        agent_executor = None
        print("\n🤖 基础聊天模式已启动（无MCP工具），输入 'quit' 退出")

    while True:
        user_input = input("\n你: ").strip()
        if user_input.lower() == "quit":
            break
        try:
            print("\nAI: ", end="", flush=True)  # 开始AI回复，不换行
            
            if agent_executor:
                # 使用Agent执行（有工具）
                full_response = ""
                async for chunk in agent_executor.astream({"input": user_input}):
                    # 处理不同类型的chunk
                    if isinstance(chunk, dict):
                        # 查找输出内容
                        if "output" in chunk:
                            content = chunk["output"]
                            if content != full_response:  # 避免重复输出
                                new_content = content[len(full_response):]
                                print(new_content, end="", flush=True)
                                full_response = content
                        elif "agent" in chunk and "messages" in chunk["agent"]:
                            # 处理agent的中间消息
                            for message in chunk["agent"]["messages"]:
                                if hasattr(message, "content") and message.content:
                                    content = message.content
                                    if content != full_response:
                                        new_content = content[len(full_response):]
                                        print(new_content, end="", flush=True)
                                        full_response = content
                        elif "steps" in chunk:
                            # 处理步骤信息（工具调用等）
                            for step in chunk["steps"]:
                                if hasattr(step, "observation") and step.observation:
                                    # 可以选择是否显示工具调用结果
                                    pass
            else:
                # 直接使用模型（无工具）
                from langchain_core.messages import HumanMessage
                response = await model.agenerate([[HumanMessage(content=user_input)]])
                content = response.generations[0][0].text
                print(content, end="", flush=True)
            
            print()  # 换行结束当前回复
            
        except Exception as exc:
            print(f"\n⚠️  出错: {exc}")

    # 清理资源
    if mcp_client:
        try:
            await mcp_client.cleanup()
            print("🧹 MCP资源已清理")
        except:
            pass
    print("Bye!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    asyncio.run(run_chat_loop())
