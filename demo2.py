import os

from dotenv import load_dotenv
from openai import OpenAI
load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "你好，请你介绍一下你自己"},
        {"role": "system", "content": "你是乐于助人的助手，请根据用户的问题给出回答"},
    ],
)
print(response.choices[0].message.content)

