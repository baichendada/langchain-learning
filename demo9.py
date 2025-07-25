from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv(override=True)

search = TavilySearch(max_retries=5)

print(search.invoke("苹果2025WWDC发布会"))