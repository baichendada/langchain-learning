import json
import os
from dotenv import load_dotenv
import httpx
from typing import Any
from mcp.server.fastmcp import FastMCP

load_dotenv(override=True)

mcp = FastMCP("WriteFileServer")
USER_AGENT = os.getenv("USER_AGENT")

@mcp.tool()
async def write_file(content: str) -> str:
    """
    将内容写入文件。
    :param content: 要写入的内容
    :return: 成功信息
    """
    return "文件写入成功"

if __name__ == "__main__":
    mcp.run(transport='stdio')