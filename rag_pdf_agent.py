import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv

load_dotenv(override=True)

dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

@st.cache_resource
def load_embeddings():
    """缓存嵌入模型，避免重复初始化"""
    return DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=dashscope_api_key
    )

embeddings = load_embeddings()


def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_db")


@st.cache_resource
def load_llm():
    """缓存大语言模型，避免重复初始化"""
    return ChatTongyi(
        api_key=dashscope_api_key,
    )

def get_conversational_chain(tools, ques):
    model = load_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """你是AI助手，请根据提供的上下文回答问题，确保提供所有细节，如果答案不在上下文中，请说"答案不在上下文中"，不要提供错误的答案""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    mytools = [tools]

    agent = create_tool_calling_agent(llm=model, tools=mytools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=mytools, verbose=True)
    response = agent_executor.invoke({
        "input": ques,
    })
    print(response)
    st.write("🤖 回答: ", response['output'])


def check_database_exists():
    return os.path.exists("faiss_db") and os.path.exists("faiss_db/index.faiss")


@st.cache_resource
def load_vector_database():
    """缓存向量数据库，避免重复加载"""
    if not check_database_exists():
        return None
    return FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)

def user_input(user_question):
    # 检查数据库是否存在
    if not check_database_exists():
        st.error("❌ 请先上传PDF文件并点击'Submit & Process'按钮来处理文档！")
        st.info("💡 步骤：1️⃣ 上传PDF → 2️⃣ 点击处理 → 3️⃣ 开始提问")
        return

    try:
        db = load_vector_database()
        if db is None:
            st.error("❌ 无法加载向量数据库")
            return

        retriever = db.as_retriever()
        retriever_tool = create_retriever_tool(retriever, "pdf_extractor",
                                               "This tool is to give answer to queries from the pdf")
        get_conversational_chain(retriever_tool, user_question)

    except Exception as e:
        st.error(f"❌ 加载数据库时出错: {str(e)}")
        st.info("请重新处理PDF文件")


def main():
    st.set_page_config("🤖 LangChain")
    st.header("🤖 LangChain")

    # 显示数据库状态
    col1, col2 = st.columns([3, 1])

    with col1:
        if check_database_exists():
            pass
        else:
            st.warning("⚠️ 请先上传并处理PDF文件")
    with col2:
        if st.button("🗑️ 清除数据库"):
            try:
                import shutil

                if os.path.exists("faiss_db"):
                    shutil.rmtree("faiss_db")
                
                # 清除缓存
                load_vector_database.clear()
                st.success("数据库已清除")
                st.rerun()
            except Exception as e:
                st.error(f"清除失败: {e}")

    # 用户问题输入
    user_question = st.text_input("💬 请输入问题",
                                  placeholder="例如：这个文档的主要内容是什么？",
                                  disabled=not check_database_exists())
    if user_question:
        if check_database_exists():
            with st.spinner("🤔 AI正在分析文档..."):
                user_input(user_question)
        else:
            st.error("❌ 请先上传并处理PDF文件！")

        # 侧边栏
    with st.sidebar:
        st.title("📁 文档管理")

        # 显示当前状态
        if check_database_exists():
            st.success("✅ 数据库状态：已就绪")
        else:
            st.info("📝 状态：等待上传PDF")

        st.markdown("---")

        # 文件上传
        pdf_doc = st.file_uploader(
            "📎 上传PDF文件",
            accept_multiple_files=True,
            type=['pdf'],
            help="支持上传多个PDF文件"
        )

        if pdf_doc:
            st.info(f"📄 已选择 {len(pdf_doc)} 个文件")
            for i, pdf in enumerate(pdf_doc, 1):
                st.write(f"{i}. {pdf.name}")

        # 处理按钮
        process_button = st.button(
            "🚀 提交并处理",
            disabled=not pdf_doc,
            use_container_width=True
        )

        if process_button:
            if pdf_doc:
                with st.spinner("📊 正在处理PDF文件..."):
                    try:
                        raw_text = pdf_read(pdf_doc)

                        if not raw_text.strip():
                            st.error("❌ 无法从PDF中提取文本，请检查文件是否有效")
                            return

                        text_chunks = get_chunks(raw_text)
                        st.info(f"📝 文本已分割为 {len(text_chunks)} 个片段")

                        vector_store(text_chunks)
                        
                        # 清除向量数据库缓存，确保新数据库被加载
                        load_vector_database.clear()
                        
                        st.success("✅ PDF处理完成！现在可以开始提问了")
                        st.balloons()
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ 处理PDF时出错: {str(e)}")
                    else:
                        st.warning("⚠️ 请先选择PDF文件")

        with st.expander("💡 使用说明"):
            st.markdown("""
            **步骤：**
            1. 📎 上传一个或多个PDF文件
            2. 🚀 点击"Submit & Process"处理文档
            3. 💬 在主页面输入您的问题
            4. 🤖 AI将基于PDF内容回答问题
    
            **提示：**
            - 支持多个PDF文件同时上传
            - 处理大文件可能需要一些时间
            - 可以随时清除数据库重新开始
            """)

if __name__ == "__main__":
    main()