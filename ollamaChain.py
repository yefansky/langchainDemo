import logging
import os
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain.schema.runnable import RunnableSequence

# 配置日志
def setup_logging(debug_mode):
    """设置日志记录"""
    if debug_mode:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_text_config(file_path):
    """读取纯文本配置文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        logging.error(f"配置文件未找到: {file_path}")
        raise
    except Exception as e:
        logging.error(f"读取配置文件时发生错误: {e}")
        raise

def setup_system(character_config):
    """基础系统设定"""
    return f"""
    你现在要扮演一个角色，你不再是AI，不管提问你任何问题都不要提及你是一个AI或现代信息。
    请保持你的语言风格，仔细学习这个角色的设定，按照要求扮演好你的角色。
    记住，你不再是AI而是一个人。
    你要扮演的角色设定：
    {character_config}
    """

def init_llm():
    """初始化 LLM"""
    try:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://10.11.39.58:31084/")
        model_name = os.getenv("OLLAMA_MODEL_NAME", "deepseek-v2")
        return Ollama(base_url=base_url, model=model_name)
    except Exception as e:
        logging.error(f"LLM初始化失败: {e}")
        raise

def create_conversation_chain(llm):
    """创建对话链"""
    conversation_prompt_template = PromptTemplate(
        input_variables=["history", "input", "system_setup"],
        template="""
        你正在扮演一个角色。根据以下对话历史和用户输入，给出合适的回复。
        角色设定：{system_setup}
        前情提要：{history}
        用户输入：{input}
        你的回复："""
    )
    return conversation_prompt_template | llm

def create_review_chain(llm):
    """创建审查链"""
    review_prompt_template = PromptTemplate(
        input_variables=["response", "review_rules"],
        template="审查规则：{review_rules}\n\n根据这些规则，请审查并修改以下回复：\n{response}\n\n如果符合规则请原样输出，如果不符合规则输出 ..."
    )
    return review_prompt_template | llm

def extract_role_name(llm, character_config):
    """使用 LLM 从角色配置中提取角色名"""
    role_name_prompt = PromptTemplate(
        input_variables=["character_config"],
        template="根据以下角色设定，提取角色的名称：\n{character_config}\n\n角色名称："
    )
    prompt = role_name_prompt.format(character_config=character_config)
    try:
        result = llm.invoke(prompt)
        return result.strip()
    except Exception as e:
        logging.error(f"提取角色名时发生错误: {e}")
        raise

def chat(debug_mode=False):
    """定义聊天过程"""
    setup_logging(debug_mode)
    
    character_config_path = 'botconfig/character_config.txt'
    review_rules_config_path = 'botconfig/review_rules.txt'

    # 加载配置
    character_config = load_text_config(character_config_path)
    review_rules_config = load_text_config(review_rules_config_path)
    
    # 初始化 LLM
    llm = init_llm()
    
    # 提取角色名称
    role_name = extract_role_name(llm, character_config)
    logging.info(f"提取的角色名称: {role_name}")
    
    # 初始化记忆
    memory = ConversationSummaryMemory(llm=llm)
    
    # 创建对话和审查链
    conversation_chain = create_conversation_chain(llm)
    review_chain = create_review_chain(llm)
    
    memory_summary = ""  # 用于存储上下文

    while True:
        try:
            # 获取用户输入
            human_input = input("你: ")
            if human_input.lower() in ["exit", "quit", "结束"]:
                break

            # 组合输入内容
            inputs = {"history": memory_summary, "input": human_input, "system_setup": setup_system(character_config)}
            logging.debug(f"对话链输入: {inputs}")
            
            # 获取机器人回答
            response = conversation_chain.invoke(inputs)
            logging.debug(f"生成的回复: {response}")
            
            # 组合审查输入
            review_inputs = {
                "response": response,
                "review_rules": review_rules_config
            }
            
            # 审查阶段
            reviewed_response = review_chain.invoke(review_inputs)
            logging.debug(f"审查后的回复: {reviewed_response}")

            # 输出最终结果
            print(f"{role_name}: {reviewed_response}")

            # 更新记忆摘要
            memory_summary += f"\n{human_input}\n{reviewed_response}"
        
        except Exception as e:
            logging.error(f"发生错误：{e}")

# 启动聊天
if __name__ == '__main__':
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    chat(debug_mode=debug_mode)
