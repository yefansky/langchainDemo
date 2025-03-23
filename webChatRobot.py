from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup

# 初始化 LLM
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="TheBloke/deepseek-llm-7B-chat-GGUF"
)

# 设置对话记忆
memory = ConversationBufferMemory()

# 自定义 Prompt 模板
prompt_template = """
你是一个智能助手，会在每次回答前通过联网搜索相关资料，整理出 top5 的信息，然后综合总结后回答用户的问题。
以下是过往的对话记录：
{history}

用户当前的问题是：
{input}

请按照以下步骤回答：
1. 根据用户的问题，在网络上搜索相关信息。
2. 从搜索结果中提取前5个最相关的信息点。
3. 综合这些信息，给出清晰、准确的回答。
"""
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=prompt_template
)

# 初始化对话链
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=False  # 设置为 True 可查看详细日志
)

# 联网搜索并提取信息的函数
def search_and_extract(query, num_results=5):
    search_results = []
    try:
        # 使用 duckduckgo_search 进行搜索
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
            for result in results:
                url = result['href']
                snippet = result['body']
                try:
                    # 获取网页内容
                    response = requests.get(url, timeout=5)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    # 提取正文（简单取 <p> 标签内容）
                    paragraphs = soup.find_all('p')
                    text = " ".join([p.get_text() for p in paragraphs[:2]])  # 取前两个段落
                    if text.strip():
                        search_results.append(f"来源: {url}\n信息: {text[:200]}...")
                    else:
                        search_results.append(f"来源: {url}\n信息: {snippet[:200]}...")
                except Exception:
                    search_results.append(f"来源: {url}\n信息: {snippet[:200]}...")
    except Exception as e:
        print(f"搜索出错: {e}")
    return search_results[:num_results]

# 主对话函数
def chat_with_bot(user_input):
    # 1. 根据用户输入进行联网搜索
    print(f"正在搜索: {user_input}")
    top5_results = search_and_extract(user_input)
    
    # 2. 将搜索结果整理为字符串
    search_summary = "以下是从网络搜索得到的 top5 信息：\n"
    for i, result in enumerate(top5_results, 1):
        search_summary += f"{i}. {result}\n"
    print(search_summary)
    
    # 3. 将用户输入和搜索结果一起传入对话链
    response = conversation.predict(input=f"{user_input}\n\n搜索结果参考:\n{search_summary}")
    
    # 4. 返回回答
    return response

# 测试多轮对话
def main():
    print("欢迎与我聊天！输入 '退出' 以结束对话。")
    while True:
        user_input = input("你: ")
        if user_input.lower() == "退出":
            print("再见！")
            break
        response = chat_with_bot(user_input)
        print(f"我: {response}")

if __name__ == "__main__":
    main()