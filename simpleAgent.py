from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, initialize_agent
from langchain.tools import tool
from langchain.agents import AgentType
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from duckduckgo_search import DDGS

# 初始化 LLM
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="TheBloke/deepseek-llm-7B-chat-GGUF"
)

# 设置对话记忆
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# 使用装饰器定义 Tool 1: 网络搜索工具
@tool
def web_search(query: str) -> str:
    """用于联网搜索信息，返回 top5 搜索结果。输入搜索关键词即可。"""
    search_results = []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            for result in results:
                url = result['href']
                snippet = result['body']
                try:
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                    }
                    response = requests.get(url, headers=headers, timeout=5)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    paragraphs = soup.find_all('p')
                    text = " ".join([p.get_text() for p in paragraphs[:2]])[:200]
                    if text.strip():
                        search_results.append(f"来源: {url}\n信息: {text}...")
                    else:
                        search_results.append(f"来源: {url}\n信息: {snippet[:200]}...")
                except Exception:
                    search_results.append(f"来源: {url}\n信息: {snippet[:200]}...")
    except Exception as e:
        return f"搜索出错: {e}"
    return "\n".join(search_results[:5]) if search_results else "未找到相关信息"

# 使用装饰器定义 Tool 2: 获取当前日期时间工具
@tool
def get_current_datetime(query: str = "") -> str:
    """获取当前日期和时间，返回格式为 'YYYY年MM月DD日 HH:MM:SS'。无需输入参数。"""
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

# 定义工具列表
tools = [web_search, get_current_datetime]

# 自定义 Prompt 模板（优化并强化格式要求）
agent_prompt = PromptTemplate.from_template("""
你是一个会使用工具的智能助手

以下是过往的对话记录：
{chat_history}

用户当前的问题是：
{input}
              
开始回答（严格按照格式输出）：
""")

# 初始化 Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# 创建 AgentExecutor 并启用错误处理
agent_executor = AgentExecutor(
    agent=agent.agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    max_execution_time=60
)

# 主对话函数（添加后处理逻辑）
def chat_with_agent(user_input):
    try:
        response = agent_executor.invoke({"input": user_input})["output"]
        # 如果响应为空或不符合格式，进行后处理
        if not response or "Final Answer" not in response:
            return "Final Answer:" + response
        return response
    except Exception as e:
        return f"Thought:执行过程中发生错误\nFinal Answer: 发生错误: {str(e)}\n"

# 测试多轮对话
def main():
    print("欢迎与我聊天！输入 '退出' 以结束对话。")
    while True:
        user_input = input("你: ")
        if user_input.lower() == "退出":
            print("再见！")
            break
        response = chat_with_agent(user_input)
        print(f"我: {response}")

if __name__ == "__main__":
    main()