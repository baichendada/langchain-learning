import os
from dotenv import load_dotenv
from langchain.output_parsers import BooleanOutputParser, ResponseSchema, StructuredOutputParser
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.chat_models import init_chat_model
import langchain_core.output_parsers
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# 从 .env 文件中加载环境变量，override=True 表示如果环境中已存在同名变量，则用 .env 文件中的值覆盖
load_dotenv(override=True)

# MODEL_API_KEY=os.getenv("DASHSCOPE_API_KEY")
# MODEL_API_KEY=os.getenv("DEEPSEEK_API_KEY")
# print(MODEL_API_KEY)

# model = init_chat_model(
#     model="deepseek-chat",
#     api_key=MODEL_API_KEY,
# )

# print(model)

# response = model.invoke("你好，请你介绍一下你自己")
# print(response)

# prompt_template = ChatPromptTemplate([
#     ("system", "你是一个乐于助人的AI助手"),
#     ("user", "这是用户的问题：{topic}， 请用 yes 或 no 来进行回答")
# ])

prompt = PromptTemplate.from_template(
    "请根据以下内容提取用户信息，并返回 JSON 格式：\n {input} \n\n{format_instructions}"
)

model = ChatTongyi(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-turbo",
)

schemas = [
    ResponseSchema(name="name", description="用户的姓名"),
    ResponseSchema(name="age", description="用户的年龄"),
]

paser = StructuredOutputParser.from_response_schemas(schemas)

print(prompt)

# basic_qa_chain = model | StrOutputParser()
# bool_qa_chain = prompt_template | model | StrOutputParser()
# bool_qa_chain = prompt_template | model | BooleanOutputParser()
chain = (
        prompt.partial(format_instructions=paser.get_format_instructions())
         | model
         | paser)
print(chain)
# response = model.invoke("你好，请你介绍一下你自己")
# response = basic_qa_chain.invoke("你好，请你介绍一下你自己")
# response = bool_qa_chain.invoke("请问 1 + 1 = 2 吗？")
response = chain.invoke({"input": "我叫baichen，今年22岁，是一名工程师"})

# print(response.content)
# print(response.additional_kwargs)
print(response)

