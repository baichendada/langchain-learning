import os

from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv(override=True)

search = TavilySearch(tavily_api_key=os.getenv("TAVILY_API_KEY"))

print(search.invoke("苹果2025WWDC发布会"))