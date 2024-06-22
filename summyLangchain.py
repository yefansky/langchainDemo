from DeepSeekLLM import DeepSeekLLM
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

llm = DeepSeekLLM(model_path=R"H:\LLM\TheBloke\deepseek-llm-7B-chat-GGUF\deepseek-llm-7b-chat.Q3_K_S.gguf", user_system_prompt="请用中文思考，用中文回答：", n_ctx = 1024, n_gpu_layers=-1, offload_kqv=True, flash_attn=True)

# 创建总结链
chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

# 执行总结链，（为了快速演示，只总结前5段）
output = chain.invoke(split_documents[:15])

print(output['output_text'])