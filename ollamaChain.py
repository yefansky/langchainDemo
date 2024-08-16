import logging
import os
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory

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

def create_sensitivity_filter(llm, sensitivity_rules):
    """创建敏感话题过滤器"""
    filter_prompt_template = PromptTemplate(
        input_variables=["text", "sensitivity_rules"],
        template="""
        以下是敏感话题的描述：
        {sensitivity_rules}

        请根据这些描述判断以下文本是否涉及敏感话题，违反其中任何一条都视做敏感话题：
        文本：{text}

        请返回一个置信度分数（0-1）表示文本是否涉及敏感话题。如果置信度高于0.5，则说明涉及敏感话题。"""
    )
    
    def filter_sensitive_text(text):
        prompt = filter_prompt_template.format(text=text, sensitivity_rules=sensitivity_rules)
        try:
            result = llm.invoke(prompt)
            confidence = float(result.strip())
            if confidence > 0.5:
                return True
            return False
        except Exception as e:
            logging.error(f"敏感话题检测发生错误: {e}")
            return False

    return filter_sensitive_text

def create_review_chain(llm, review_rules):
    """创建文本审查链"""
    review_prompt_template = PromptTemplate(
        input_variables=["text", "review_rules"],
        template="""
        以下是审查规则：
        {review_rules}

        请根据这些规则润色以下文本：
        文本：{text}

        请返回润色后的文本。"""
    )
    
    def review_text(text):
        prompt = review_prompt_template.format(text=text, review_rules=review_rules)
        try:
            result = llm.invoke(prompt)
            return result.strip()
        except Exception as e:
            logging.error(f"文本审查发生错误: {e}")
            return text

    return review_text

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
    sensitivity_rules_path = 'botconfig/sensitivity_rules.txt'
    review_rules_path = 'botconfig/review_rules.txt'

    # 加载配置
    character_config = load_text_config(character_config_path)
    sensitivity_rules = load_text_config(sensitivity_rules_path)
    review_rules = load_text_config(review_rules_path)
    
    # 初始化 LLM
    llm = init_llm()
    
    # 提取角色名称
    role_name = extract_role_name(llm, character_config)
    logging.info(f"提取的角色名称: {role_name}")
    
    # 初始化记忆
    memory = ConversationSummaryMemory(llm=llm)
    
    # 创建对话链
    conversation_chain = create_conversation_chain(llm)
    
    # 创建敏感话题过滤器
    sensitivity_filter = create_sensitivity_filter(llm, sensitivity_rules)
    
    # 创建文本审查链
    review_chain = create_review_chain(llm, review_rules)
    
    memory_summary = ""  # 用于存储上下文
    default_reply = "对不起，这个话题不在我的讨论范围之内。"

    while True:
        try:
            # 获取用户输入
            human_input = input("你: ")
            if human_input.lower() in ["exit", "quit", "结束"]:
                break

            # 过滤用户输入中的敏感话题
            if sensitivity_filter(human_input):
                response = default_reply
            else:
                # 组合输入内容
                inputs = {"history": memory_summary, "input": human_input, "system_setup": setup_system(character_config)}
                logging.debug(f"对话链输入: {inputs}")
                
                # 获取机器人回答
                response = conversation_chain.invoke(inputs)
                logging.debug(f"生成的回复: {response}")
                
                # 过滤LLM回复中的敏感话题
                if sensitivity_filter(response):
                    response = default_reply
                    logging.info("LLM回复中包含敏感话题，使用默认回复。")
                
                # 润色文本
                response = review_chain(response)
                
            # 输出最终结果
            print(f"{role_name}: {response}")

            # 更新记忆摘要（仅在回复不包含敏感话题时）
            if response != default_reply:
                memory_summary += f"\n{human_input}\n{response}"
        
        except Exception as e:
            logging.error(f"发生错误：{e}")

# 启动聊天
if __name__ == '__main__':
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    chat(debug_mode=debug_mode)
