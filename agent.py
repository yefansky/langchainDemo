import json
import re
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import FAISS
from langchain.prompts import MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
import os
import uuid
from typing import List, Dict, Optional
from fnmatch import fnmatch

# 配置
PAGE_LIMIT = 4000
MEMORY_DIR = "./external_memory"
if not os.path.exists(MEMORY_DIR):
    os.makedirs(MEMORY_DIR)

# DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-95cd174acb2d4710b410342bee3b58b6"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
faiss_path = os.path.join(MEMORY_DIR, "faiss_index")
if os.path.exists(faiss_path):
    vector_store = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
else:
    vector_store = FAISS.from_texts(["初始记忆"], embeddings, metadatas=[{"id": str(uuid.uuid4())}])
    vector_store.save_local(faiss_path)

# 分页函数
def paginate_text(text: str, page: int = 1, file_path: Optional[str] = None) -> str:
    is_code = False
    if file_path:
        code_extensions = {".cpp", ".h", ".cs", ".py", ".js"}
        is_code = any(file_path.endswith(ext) for ext in code_extensions)
    else:
        code_keywords = {"def ", "function ", "class ", "void ", "int ", "if ("}
        is_code = any(keyword in text[:100] for keyword in code_keywords)

    if len(text) <= PAGE_LIMIT:
        return json.dumps({"content": text, "page": 1, "total_pages": 1, "remaining_pages": 0}, ensure_ascii=False)

    if is_code:
        language = Language.PYTHON if file_path and file_path.endswith(".py") else Language.CPP
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=PAGE_LIMIT,
            chunk_overlap=0
        )
        pages = splitter.split_text(text)
    else:
        lines = text.splitlines(keepends=True)
        pages = []
        current_page = ""
        for line in lines:
            if len(current_page) + len(line) <= PAGE_LIMIT:
                current_page += line
            else:
                if current_page:
                    pages.append(current_page)
                current_page = line if len(line) <= PAGE_LIMIT else line[:PAGE_LIMIT]
        if current_page:
            pages.append(current_page)

    total_pages = len(pages)
    if page > total_pages or page < 1:
        return json.dumps({"error": f"页码 {page} 超出范围，总页数 {total_pages}"}, ensure_ascii=False)
    return json.dumps({
        "content": pages[page - 1],
        "page": page,
        "total_pages": total_pages,
        "remaining_pages": total_pages - page
    }, ensure_ascii=False)

# 记忆管理
memory_prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是一个记忆管理助手，任务是总结任务输出并管理外部记忆。
当前任务输入：{input}
任务输出：{output}
召回的相关记忆（Top-3）：
{retrieved_memory}

任务：
1. 总结当前任务输出，生成新记忆。
2. 根据召回的记忆，关联总结并决定是否修改已有记忆（每条记忆有唯一 ID）。
3. 输出新记忆内容和可能的修改指令。

输出格式（JSON 字符串）：
{"type": "new_memory", "content": "新记忆内容"}
或
{"type": "modify_memory", "id": "ID", "content": "新内容"}
或
{"type": "no_change"}
""")
])

memory_llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    temperature=0,
    openai_api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

def process_memory(input_text: str, output_text: str) -> str:
    docs = vector_store.similarity_search(input_text + " " + output_text, k=3)
    retrieved_memory = "\n".join([f"ID: {doc.metadata['id']}, Content: {doc.page_content}" for doc in docs])
    
    memory_response = memory_prompt | memory_llm | StrOutputParser()
    result = memory_response.invoke({"input": input_text, "output": output_text, "retrieved_memory": retrieved_memory})
    
    new_memory = None
    modify_instructions = []
    lines = result.splitlines()
    for line in lines:
        try:
            data = json.loads(line)
            if data["type"] == "new_memory":
                new_memory = data["content"]
            elif data["type"] == "modify_memory":
                modify_instructions.append({"id": data["id"], "content": data["content"]})
        except json.JSONDecodeError:
            return json.dumps({"error": f"记忆管理输出无效: {line}"}, ensure_ascii=False)
    
    if new_memory:
        new_id = str(uuid.uuid4())
        vector_store.add_texts([new_memory], metadatas=[{"id": new_id}])
    
    for instr in modify_instructions:
        vector_store.delete([instr["id"]])
        vector_store.add_texts([instr["content"]], metadatas=[{"id": instr["id"]}])
    
    vector_store.save_local(faiss_path)
    summary = f"已处理记忆，新增: {bool(new_memory)}, 修改: {len(modify_instructions)} 条"
    return paginate_text(summary)

# 工具定义
@tool
def scan_directory(directory: str, page: int = 1) -> str:
    """扫描指定目录，返回代码文件路径列表（分页，每页不超过4k字符）。
    考虑 .gitignore 规则，并屏蔽以 '.' 开头的目录（如 .vscode、.git）。

    Args:
        directory (str): 要扫描的目录路径。
        page (int): 返回结果的页码，默认为 1。

    Returns:
        str: JSON 格式的分页文件路径列表，或错误信息。
    """
    if not os.path.exists(directory):
        return json.dumps({"error": f"目录 {directory} 不存在"}, ensure_ascii=False)

    # 读取 .gitignore 文件（如果存在）
    ignore_patterns = []
    gitignore_path = os.path.join(directory, ".gitignore")
    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):  # 忽略空行和注释
                    ignore_patterns.append(line)

    code_files = []
    for root, dirs, files in os.walk(directory, topdown=True):
        # 获取当前目录的相对路径
        rel_root = os.path.relpath(root, directory)

        # 跳过以 '.' 开头的目录（如 .vscode、.git）
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        # 检查当前目录是否被 .gitignore 忽略
        if rel_root != ".":
            rel_root_path = rel_root + os.sep
            if any(fnmatch(rel_root_path, pattern) or fnmatch(rel_root, pattern) for pattern in ignore_patterns):
                continue

        for file in files:
            # 只处理指定代码文件类型
            if file.endswith((".cpp", ".h", ".cs", ".py", ".js")):
                file_path = os.path.join(rel_root, file) if rel_root != "." else file
                full_path = os.path.join(root, file)

                # 检查文件是否被 .gitignore 忽略
                ignored = False
                for pattern in ignore_patterns:
                    # 处理目录和文件模式
                    if pattern.endswith("/"):
                        if fnmatch(file_path + "/", pattern):
                            ignored = True
                            break
                    elif fnmatch(file_path, pattern) or fnmatch(file, pattern):
                        ignored = True
                        break

                if not ignored:
                    code_files.append(full_path)

    if not code_files:
        return json.dumps({"content": "没有找到符合条件的代码文件", "page": 1, "total_pages": 1, "remaining_pages": 0}, ensure_ascii=False)

    full_text = "\n".join(code_files)
    return paginate_text(full_text, page)

@tool
def read_file(file_path: str, page: int = 1) -> str:
    """读取指定文件的内容（分页，每页不超过4k字符）。
    
    Args:
        file_path (str): 要读取的文件路径。
        page (int): 返回结果的页码，默认为 1。
    
    Returns:
        str: JSON 格式的分页文件内容，或错误信息。
    """
    if not os.path.exists(file_path):
        return json.dumps({"error": f"文件 {file_path} 不存在"}, ensure_ascii=False)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return json.dumps({"error": f"读取文件 {file_path} 失败: {str(e)}"}, ensure_ascii=False)
    return paginate_text(content, page, file_path=file_path)

@tool
def search_full_text(file_list: List[str], keyword: str, max_results: int = 5, page: int = 1, mode: str = "normal") -> str:
    """在多个文件中搜索关键词，支持普通模式和正则表达式模式（分页，每页不超过4k字符）。
    
    Args:
        file_list (List[str]): 要搜索的文件路径列表。
        keyword (str): 要搜索的关键词。
        max_results (int): 最大返回结果数，默认为 5。
        page (int): 返回结果的页码，默认为 1。
        mode (str): 搜索模式，'normal' 或 'regex'，默认为 'normal'。
    
    Returns:
        str: JSON 格式的分页搜索结果，或错误信息。
    """
    results = []
    for file_path in file_list:
        if not os.path.exists(file_path):
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()
                for i, line in enumerate(lines, 1):
                    if mode == "normal":
                        if keyword in line:
                            results.append(f"{file_path}:{i}:{line.strip()}")
                    elif mode == "regex":
                        try:
                            if re.search(keyword, line):
                                results.append(f"{file_path}:{i}:{line.strip()}")
                        except re.error:
                            return json.dumps({"error": f"无效的正则表达式: {keyword}"}, ensure_ascii=False)
                    else:
                        return json.dumps({"error": f"不支持的模式: {mode}，支持 'normal' 或 'regex'"}, ensure_ascii=False)
                    if len(results) >= max_results * 10:
                        break
        except Exception as e:
            return json.dumps({"error": f"读取文件 {file_path} 失败: {str(e)}"}, ensure_ascii=False)
    full_text = "\n".join(results)
    return paginate_text(full_text, page)

@tool
def observe_context(file_path: str, line_number: int, context_lines: int = 5, page: int = 1) -> str:
    """根据文件路径和行号提取上下文（分页，每页不超过4k字符）。
    
    Args:
        file_path (str): 要读取的文件路径。
        line_number (int): 目标行号。
        context_lines (int): 上下上下文行数，默认为 5。
        page (int): 返回结果的页码，默认为 1。
    
    Returns:
        str: JSON 格式的分页上下文内容，或错误信息。
    """
    if not os.path.exists(file_path):
        return json.dumps({"error": f"文件 {file_path} 不存在"}, ensure_ascii=False)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        return json.dumps({"error": f"读取文件 {file_path} 失败: {str(e)}"}, ensure_ascii=False)
    start = max(0, line_number - context_lines - 1)
    end = min(len(lines), line_number + context_lines)
    context = "".join(lines[start:end]).strip()
    return paginate_text(context, page, file_path=file_path)

@tool
def manage_memory(input_text: str, output_text: str) -> str:
    """管理外部记忆，总结输入和输出并存入或修改 RAG。
    
    Args:
        input_text (str): 输入文本。
        output_text (str): 输出文本。
    
    Returns:
        str: JSON 格式的记忆管理结果。
    """
    return process_memory(input_text, output_text)

tools = [scan_directory, read_file, search_full_text, observe_context, manage_memory]

# Agent Prompt（显式避免 "type" 作为输入变量）
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是一个智能助手，能够处理多种任务，包括但不限于代码分析、文档处理等。
当前任务状态存储在 {state} 中。
召回的外部记忆（Top-3）：
{retrieved_memory}

你的目标是根据用户输入，灵活规划任务执行步骤，并利用提供的工具推进工作。
工具的用法和参数格式已绑定到模型中，你可以根据需要调用它们。
任务执行可能包括：
- 制定单层计划（仅当前任务的步骤）
- 调用工具获取信息
- 根据观察到的信息推进或调整当前计划，或生成子任务计划
- 输出当前任务的结果

所有输出必须为 JSON 格式字符串，支持以下类型：
- 计划：{{"type": "plan", "steps": ["步骤1", "步骤2", ...]}}（仅单层）
- 工具调用：{{"type": "tool", "name": "工具名", "args": {{参数字典}}}}
- 观察：{{"type": "observation", "content": "观察到的信息"}}
- 调整计划：{{"type": "adjust_plan", "steps": ["新步骤1", "新步骤2", ...]}}（仅修改当前层）
- 结果：{{"type": "result", "content": "当前任务结果"}}
- 反思：{{"type": "reflection", "content": "反思内容"}}

根据输入和当前状态，自行决定下一步行动。
"""),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Agent 管道
state = {
    "files": [],
    "search_results": [],
    "context": [],
    "reflection_count": 0
}

def retrieve_top_k_memory(input_dict: Dict) -> Dict:
    query = input_dict["input"]
    docs = vector_store.similarity_search(query, k=3)
    retrieved_memory = "\n".join([f"ID: {doc.metadata['id']}, Content: {doc.page_content}" for doc in docs])
    input_dict["state"] = state
    input_dict["retrieved_memory"] = retrieved_memory
    return input_dict

def invoke_tools(response):
    if isinstance(response, AIMessage) and "tool_calls" in response.additional_kwargs:
        tool_calls = response.additional_kwargs["tool_calls"]
        for call in tool_calls:
            tool_name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"])  # 改为 json.loads 以避免 eval 的安全风险
            tool = next(t for t in tools if t.name == tool_name)
            result = tool.invoke(args)
            result_dict = json.loads(result)
            if "error" in result_dict:
                return result
            if tool_name == "scan_directory":
                state["files"].append(result_dict)
            elif tool_name == "search_full_text":
                state["search_results"].append(result_dict)
            elif tool_name in ["read_file", "observe_context"]:
                state["context"].append(result_dict)
            return result
    else:
        try:
            result_dict = json.loads(response.content)
            if result_dict["type"] == "reflection":
                state["reflection_count"] += 1
            return response.content
        except json.JSONDecodeError:
            return json.dumps({"error": f"无效输出: {response.content}"}, ensure_ascii=False)

llm = ChatOpenAI(
    model=DEEPSEEK_MODEL,
    temperature=0,
    openai_api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

agent_chain = (
    RunnableLambda(retrieve_top_k_memory)
    | agent_prompt
    | llm.bind_tools(tools)
    | RunnableLambda(invoke_tools)
    | StrOutputParser()
)

# 运行逻辑
def run_agent_with_memory(initial_input: str):
    input_dict = {"input": initial_input}
    task_stack = []  # 任务栈：{"plan": List[str], "current_step": int, "observations": List[str]}
    all_results = []  # 存储所有任务结果
    agent_scratchpad = []  # 存储对话历史

    while True:
        # 添加 agent_scratchpad 到输入
        input_dict["agent_scratchpad"] = agent_scratchpad
        result = agent_chain.invoke(input_dict)
        print(result)
        try:
            result_dict = json.loads(result)

            if result_dict["type"] == "plan":
                task_stack.append({
                    "plan": result_dict["steps"],
                    "current_step": 0,
                    "observations": []
                })
                process_memory(initial_input, result)
                agent_scratchpad.append(AIMessage(content=result))
                if task_stack:
                    current_task = task_stack[-1]
                    input_dict["input"] = f"执行计划步骤: {current_task['plan'][current_task['current_step']]}"

            elif result_dict["type"] == "tool":
                process_memory(initial_input, result)
                agent_scratchpad.append(AIMessage(content=result))
                if task_stack:
                    current_task = task_stack[-1]
                    current_task["observations"].append(f"工具调用结果: {result}")
                    input_dict["input"] = f"工具调用结果: {result}"

            elif result_dict["type"] == "observation":
                if task_stack:
                    current_task = task_stack[-1]
                    current_task["observations"].append(result_dict["content"])
                    process_memory(initial_input, result)
                    agent_scratchpad.append(AIMessage(content=result))
                    if current_task["current_step"] < len(current_task["plan"]) - 1:
                        current_task["current_step"] += 1
                        input_dict["input"] = f"根据观察 '{result_dict['content']}' 执行下一步: {current_task['plan'][current_task['current_step']]}"
                    else:
                        input_dict["input"] = f"当前计划完成，观察结果: {result_dict['content']}，请总结或生成子任务计划"

            elif result_dict["type"] == "adjust_plan":
                if task_stack:
                    current_task = task_stack[-1]
                    current_task["plan"] = result_dict["steps"]
                    current_task["current_step"] = 0
                    process_memory(initial_input, result)
                    agent_scratchpad.append(AIMessage(content=result))
                    input_dict["input"] = f"计划已调整，新步骤: {current_task['plan'][current_task['current_step']]}"

            elif result_dict["type"] == "result":
                if task_stack:
                    current_task = task_stack.pop()
                    all_results.append(result_dict["content"])
                    process_memory(initial_input, result)
                    agent_scratchpad.append(AIMessage(content=result))
                    if task_stack:  # 还有父任务
                        parent_task = task_stack[-1]
                        parent_task["observations"].append(f"子任务结果: {result_dict['content']}")
                        if parent_task["current_step"] < len(parent_task["plan"]) - 1:
                            parent_task["current_step"] += 1
                            input_dict["input"] = f"根据子任务结果 '{result_dict['content']}' 执行下一步: {parent_task['plan'][parent_task['current_step']]}"
                        else:
                            input_dict["input"] = f"父任务计划完成，子任务结果: {result_dict['content']}，请总结"
                    else:
                        break  # 所有任务完成
                else:
                    process_memory(initial_input, result)
                    agent_scratchpad.append(AIMessage(content=result))
                    break

            elif result_dict["type"] == "reflection":
                state["reflection_count"] += 1
                process_memory(initial_input, result)
                agent_scratchpad.append(AIMessage(content=result))
                if state["reflection_count"] >= 3:
                    print(json.dumps({"error": "反思次数达到上限，停止执行"}, ensure_ascii=False))
                    break
                input_dict["input"] = f"反思: {result_dict['content']}，请继续处理"

        except json.JSONDecodeError:
            print(json.dumps({"error": f"无法解析结果: {result}"}, ensure_ascii=False))
            agent_scratchpad.append(AIMessage(content=result))
            break
        except KeyError as e:
            print(f"键错误: {e}，响应: {result}")
            agent_scratchpad.append(AIMessage(content=result))
            break

    return all_results

# 测试
if __name__ == "__main__":
    results = run_agent_with_memory("分析 e:\\Code\\CodeReDesign\\ 目录下的代码，查找配置多个api的baseurl和apikey的地方在哪里?")
    print("最终结果:", results)