from langchain_community.llms import Ollama

url = "http://10.11.39.58:31084/"

llm = Ollama(
    base_url = url,
    model="deepseek-v2"
)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    result = llm.invoke("给我讲一个笑话")
    print(result)