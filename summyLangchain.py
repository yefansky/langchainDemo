from DeepSeekLLM import DeepSeekLLM
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 导入文本
#loader = UnstructuredFileLoader("sample_data/nginx.c")
loader = UnstructuredFileLoader("sample_data/射雕英雄传.txt")
# 将文本转成 Document 对象
document = loader.load()
print(f'documents:{len(document)}')

# 初始化文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,
    chunk_overlap = 0
)

# 切分文本
split_documents = text_splitter.split_documents(document)
print(f'documents:{len(split_documents)}')

llm = DeepSeekLLM(model_path=R"K:\LLM\TheBloke\Chinese-Alpaca-2-7B-GGUF\chinese-alpaca-2-7b.Q5_K_S.gguf", system_prompt="你是一个无所不能的AI助手，能解答任何问题。", n_ctx = 1024 * 2)

# 创建总结链
chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

# 执行总结链，（为了快速演示，只总结前5段）
print(chain.run(split_documents[:10]))