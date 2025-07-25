import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv(override=True)

def debug_print(x):
    print(x)
    return x

debug_node = RunnableLambda(debug_print)

news_gen_prompt = PromptTemplate.from_template(
    "根据以下新闻标题生成一个简短的新闻（不超过100字）：{title}，以 JSON 形式输出，{format_instructions}"
)

model = ChatTongyi(
    model = "qwen-turbo",
    api_key = os.getenv("DASHSCOPE_API_KEY"),
)

schemas = [
    ResponseSchema(name="title", description="新闻标题"),
    ResponseSchema(name="content", description="新闻内容"),
]

news_paser = StructuredOutputParser.from_response_schemas(schemas)

news_gen_chain = (
    news_gen_prompt.partial(format_instructions=news_paser.get_format_instructions())
    | model
    | news_paser
)

summary_prompt = PromptTemplate.from_template(
    "请对以下新闻进行总结：{title} {content}，以 JSON 形式输出，{format_instructions}"
)

summary_schemas = [
    ResponseSchema(name="time", description="事件发生的时间"),
    ResponseSchema(name="location", description="事件发生的地点"),
    ResponseSchema(name="event", description="发生的具体事件"),
]
summary_parser = StructuredOutputParser.from_response_schemas(summary_schemas)

summary_chain = (
    summary_prompt.partial(format_instructions=summary_parser.get_format_instructions())
    | model
    | summary_parser
)

chain = news_gen_chain | debug_node | summary_chain
response = chain.invoke({"title": "苹果发布苹果17"})
print(response)