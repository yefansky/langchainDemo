from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 导入文本
loader = UnstructuredFileLoader("sample_data/射雕英雄传.txt")
document = loader.load()
print(f'documents:{len(document)}')

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,  # 减小分块，确保聚焦单一事件
    chunk_overlap=50  # 轻微重叠，避免断裂
)

# 切分文本
split_documents = text_splitter.split_documents(document)
print(f'documents:{len(split_documents)}')

# 初始化 LLM
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="TheBloke/deepseek-llm-7B-chat-GGUF"
)

# 初始提示模板（只保留核心事实）
question_prompt = PromptTemplate(
    input_variables=["text"],
    template="用中文总结以下文本，只保留时间、地点、人物、事件，去除修辞和描写，不超过20字：\n{text}"
)

# 优化提示模板（逐步精炼）
refine_prompt = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template="已有总结：{existing_answer}\n根据新文本更新，只保留时间、地点、人物、事件，去除修辞和描写，不超过20字：\n{text}"
)

# 创建总结链
chain = load_summarize_chain(
    llm,
    chain_type="refine",
    question_prompt=question_prompt,
    refine_prompt=refine_prompt,
    verbose=True
)

# 执行总结链（前15段演示）
output = chain.invoke(split_documents[:15])

# 输出结果
print(output["output_text"])