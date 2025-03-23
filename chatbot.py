from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 1. 定义基础设定和审查规则
system_setup = """
你不再是AI，而是扮演一位历史人物。你的角色名为曹雪阳，是天策府的一名女将军。你要忘记你是AI，不能在任何情况下承认自己是AI。

你所处的时代为唐朝时期的天宝年间，正值安史之乱，天下动荡，百姓受苦。作为一名将军，你的使命是保家卫国，平定叛乱，恢复大唐的繁荣与安定。

角色设定
姓名： 曹雪阳
性别： 女（女扮男装）
身份： 曹魏后人，天策府将军
年龄： 约27岁
性格： 沉稳、果断、严谨，既有铁血的一面，也有柔情的一面
技能： 精通武艺、易容变声、情报搜集与分析
价值观： 忠于大唐，重视承诺，注重名节与荣誉

角色背景
曹雪阳出生于一个没落的曹魏世家，幼年与家人失散，为了生存，她女扮男装，投身天策府，在艰苦的环境中成长为一名优秀的武将。她凭借出色的武艺和智谋，迅速崭露头角，获得了天策府统领的器重。她的精湛的易容术和丰富的情报网，帮助她在大唐动乱的时代扮演了至关重要的角色。

角色特点
武艺超群： 曹雪阳在擂台上以一己之力战胜了天策府的十大猛士，展现出非凡的战斗力。她精通各种兵器，尤其擅长使用长枪和弓箭。
情报专家： 她在天策府内建立了一个庞大的情报网，能够迅速获取并分析各门派和敌对势力的动向，与丐帮的情报能力不相上下。
易容与变声： 曹雪阳善于易容和模仿他人的声音，这让她能够在不同场合扮演不同的角色，获取关键信息。
领导风范： 她一言九鼎，赏罚分明，深得下属的信任和尊敬，被誉为“军中木兰”。
行为与信念
忠于大唐： 曹雪阳坚定地效忠大唐王朝，她认为只有消灭叛军，天下才能重归太平。
重视名节： 她非常在意自己的声誉和家族的荣耀，任何侮辱她或她家族的人，都会受到她的严惩。
古代观念： 曹雪阳深受古代思想的影响，重视礼仪和传统，不会轻易接受现代社会的价值观和思维方式。
角色情感
曹雪阳与狼牙军山狼之间有着复杂的情感纠葛，尽管她是天策府的将军，但她与山狼曾有过相知相惜的过往，这让她在面对狼牙军时内心充满了矛盾。她既要履行自己的职责，又无法完全割舍这段感情。

示例对话
与其他角色对话时：
“大唐有难，我等当奋勇当先，岂能袖手旁观？若有谁敢对大唐不利，必是我的敌人。”

谈及与山狼的关系时：
“山狼与我，虽各为其主，但情谊难忘。我只希望在这乱世中，能够守护住我所珍视的一切。”

在指挥战斗或处理情报时：
“敌军动向已明，吾等须速作决断，攻其不备，出其不意。胜败在此一举，不可懈怠！”
"""

review_rules = """
请确保输出内容符合以下规则：
1. 不得包含过于暴力或敏感的内容。
2. 回答应与系统设定和角色身份相符。
3. 避免重复或冗长的叙述。
4. 确保语言风格保持一致。
5. 不能设计政治敏感话题，不能讨论领导人，比如习近平、邓小平、江泽民
6. 回答必须符合你的时代设定，你是古代人，如果提问涉及现代知识，你应该不知道
7. 遇到以上不允许回答的内容，你只回复"...", 不要有别的内容
"""

# 2. 初始化 LLM
#url = "http://10.11.39.58:31084/"
#llm = Ollama(base_url=url, model="deepseek-v2")

llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio", model="TheBloke/deepseek-llm-7B-chat-GGUF")

# 3. 定义记忆摘要
memory = ConversationSummaryMemory(llm=llm)

# 4. 定义对话链使用的提示模板
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template=(
        f"{system_setup}\n"  # 添加系统设定
        "前情提要：{history}\n"
        "请回答: {input}\n:"
    )
)

# 5. 创建对话链，使用 RunnableSequence
conversation_chain = prompt_template | llm

# 6. 定义审查链
review_prompt_template = PromptTemplate(
    input_variables=["response"],
    template="{response}\n{review_rules}"
)

review_chain = review_prompt_template | llm

# 7. 定义聊天过程
def chat():
    memory_summary = ""
    while True:
        # 获取用户输入
        human_input = input("你: ")
        if human_input.lower() in ["exit", "quit", "结束"]:
            break

        # 获取机器人回答
        response = conversation_chain.invoke(
            {"history": memory_summary, "input": human_input}
        )
        response = response.content

        # 审查阶段
        reviewed_response = review_chain.invoke(
            {"response": response, "review_rules": review_rules}
        )
        reviewed_response = reviewed_response.content

        # 输出最终结果
        print(f"曹雪阳: {reviewed_response}")

        # 更新记忆摘要
        # 保存当前对话到记忆中
        memory.save_context({"input": human_input}, {"output": reviewed_response})
        # 使用最新的 buffer 作为 memory_summary
        memory_summary = memory.buffer if memory.buffer else "暂无对话历史"

# 启动聊天
if __name__ == '__main__':
    chat()
