import os
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputKeyToolsParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_experimental.tools import PythonAstREPLTool

load_dotenv(override=True)

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

pd.set_option("max_colwidth", 200)
print(df.head(5))

df.info()

interpreter_tool = PythonAstREPLTool(locals={"df": df})
print(interpreter_tool.invoke("df['SeniorCitizen'].mean()"))
print(interpreter_tool.name)

# --------------

system = f"""
你可以访问一个名为 `df` 的 pandas 数据框。请根据用户的问题，直接编写相应的 Python 代码来计算答案。
可以使用 df.head() 或 df.info() 等查看数据的方法，执行用户要求的计算操作。
只返回执行计算的代码，不返回其他内容。只允许使用 pandas 和内置库。
"""

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(system),
        HumanMessage("{input}")
    ]
)

parser = JsonOutputKeyToolsParser(key_name=interpreter_tool.name, first_tool_only=True)

model = ChatTongyi(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-plus"
)

llm_with_tools = model.bind_tools([interpreter_tool])

# response = llm_with_tools.invoke("我有一张表，名为df，请帮我计算MonthlyCharges字段的均值。")
# print(response)

def debug_print(x):
    print(x)
    return x

tools_chain = prompt | llm_with_tools | parser | RunnableLambda(debug_print) | interpreter_tool
response = tools_chain.invoke({"input": "我有一张表，名为df，请帮我计算MonthlyCharges字段的均值。"})
print(response)

# ==================== 新的两阶段Chain实现 ====================

# 结果解释的提示词
answer_prompt = ChatPromptTemplate.from_messages([
    SystemMessage("你是一个数据分析助手。用户问了一个关于数据的问题，我们已经执行了相应的代码并得到了结果。请根据问题和结果，给出清晰的中文回答。"),
    HumanMessage("用户问题：{question}"),
    HumanMessage("代码执行结果：{tool_result}"),
    HumanMessage("请给出最终答案：")
])

str_parser = StrOutputParser()

def execute_tool_and_answer(inputs):
    """执行工具并基于结果生成最终答案"""
    question = inputs["input"]
    
    # 第一步：生成并执行工具调用
    tool_chain = prompt | llm_with_tools | parser
    tool_call = tool_chain.invoke({"input": question})
    print(f"生成的工具调用: {tool_call}")
    
    # 执行工具
    tool_result = interpreter_tool.invoke(tool_call)
    print(f"工具执行结果: {tool_result}")
    
    # 第二步：基于工具结果生成最终答案
    answer_chain = answer_prompt | model | str_parser
    final_answer = answer_chain.invoke({
        "question": question,
        "tool_result": tool_result
    })
    
    return final_answer

# 创建完整的chain
complete_chain = RunnableLambda(execute_tool_and_answer)

# 测试新的chain
print("\n========== 使用新的两阶段Chain ==========")
response = complete_chain.invoke({"input": "我有一张表，名为df，请帮我计算MonthlyCharges字段的均值。"})
print("最终答案:", response)
