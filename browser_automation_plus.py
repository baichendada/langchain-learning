from dotenv import load_dotenv
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.chat_models import ChatTongyi
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_tool_calling_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
from datetime import datetime

load_dotenv(override=True)


@tool
def summarize_website(url: str) -> str:
    """
    Summarize the content of a website.
    """
    # 创建浏览器工具
    try:
        sync_browser = create_sync_playwright_browser()
        toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser)
        tools = toolkit.get_tools()

        # 创建大模型和agent
        model = ChatTongyi(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        prompt = hub.pull("hwchase17/openai-tools-agent")
        agent = create_openai_tools_agent(model, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        # 创建指令
        command = {
            "input": f"总结这个网站中的内容: {url}",
        }
        # 执行指令
        result = agent_executor.invoke(command)
        return result.get("output", "No output found.")
    except Exception as e:
        return f"An error occurred: {str(e)}"


@tool
def generate_pdf(content: str) -> str:
    """将文本内容生成为PDF文件"""
    try:
        # 生成文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"website_summary_{timestamp}.pdf"

        # 创建PDF文档
        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()

        # 注册中文字体（如果系统有的话）
        try:
            # Windows 系统字体路径
            font_paths = [
                "C:/Windows/Fonts/simhei.ttf",  # 黑体
                "C:/Windows/Fonts/simsun.ttc",  # 宋体
                "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            ]

            chinese_font_registered = False
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                        chinese_font_registered = True
                        print(f"✅ 成功注册中文字体: {font_path}")
                        break
                    except:
                        continue

            if not chinese_font_registered:
                print("⚠️ 未找到中文字体，使用默认字体")

        except Exception as e:
            print(f"⚠️ 字体注册失败: {e}")

        # 自定义样式 - 支持中文
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            alignment=TA_CENTER,
            spaceAfter=30,
            fontName='ChineseFont' if 'chinese_font_registered' in locals() and chinese_font_registered else 'Helvetica-Bold'
        )

        content_style = ParagraphStyle(
            'CustomContent',
            parent=styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=12,
            fontName='ChineseFont' if 'chinese_font_registered' in locals() and chinese_font_registered else 'Helvetica'
        )

        # 构建PDF内容
        story = []

        # 标题
        story.append(Paragraph("网站内容总结报告", title_style))
        story.append(Spacer(1, 20))

        # 生成时间
        time_text = f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(time_text, styles['Normal']))
        story.append(Spacer(1, 20))

        # 分隔线
        story.append(Paragraph("=" * 50, styles['Normal']))
        story.append(Spacer(1, 15))

        # 主要内容 - 改进中文处理
        if content:
            # 清理和处理内容
            content = content.replace('\r\n', '\n').replace('\r', '\n')
            paragraphs = content.split('\n')

            for para in paragraphs:
                if para.strip():
                    # 处理特殊字符，确保PDF可以正确显示
                    clean_para = para.strip()
                    # 转换HTML实体
                    clean_para = clean_para.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')

                    try:
                        story.append(Paragraph(clean_para, content_style))
                        story.append(Spacer(1, 8))
                    except Exception as para_error:
                        # 如果段落有问题，尝试用默认字体
                        try:
                            fallback_style = ParagraphStyle(
                                'Fallback',
                                parent=styles['Normal'],
                                fontSize=10,
                                leftIndent=20,
                                rightIndent=20,
                                spaceAfter=10
                            )
                            story.append(Paragraph(clean_para, fallback_style))
                            story.append(Spacer(1, 8))
                        except:
                            # 如果还是有问题，记录错误但继续
                            print(f"⚠️ 段落处理失败: {clean_para[:50]}...")
                            continue
        else:
            story.append(Paragraph("暂无内容", content_style))

        # 页脚信息
        story.append(Spacer(1, 30))
        story.append(Paragraph("=" * 50, styles['Normal']))
        story.append(Paragraph("本报告由 Playwright PDF Agent 自动生成", styles['Italic']))

        # 生成PDF
        doc.build(story)

        # 获取绝对路径
        abs_path = os.path.abspath(filename)
        print(f"📄 PDF文件生成完成: {abs_path}")
        return f"PDF文件已成功生成: {abs_path}"

    except Exception as e:
        error_msg = f"PDF生成失败: {str(e)}"
        print(error_msg)
        return error_msg


model = ChatTongyi(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    model="qwen-turbo",
)

prompt = ChatPromptTemplate.from_template(
    """请优化以下网站总结内容，使其更适合PDF报告格式：

    原始总结：
    {summary}

    请重新组织内容，包括：
    1. 清晰的标题和结构
    2. 要点总结
    3. 详细说明
    4. 使用要求等

    优化后的内容："""
)


chain = (
    summarize_website
    | (lambda summary: {"summary": summary})
    | prompt
    | model
    | StrOutputParser()
    | generate_pdf
)

chain.invoke({
    "url": "https://blog.csdn.net/baichendada/article/details/148673714?spm=1001.2014.3001.5501"
})