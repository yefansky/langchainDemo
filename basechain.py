from llama_cpp import Llama
from langchain_core.language_models.llms import LLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import Any, Dict, Iterator, List, Mapping, Optional

class LlamaWrapper:
    def __init__(self, model_path, n_gpu_layers=-1, seed=1337, n_ctx=2048, chat_format="chatml", verbose=False):
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            n_ctx=n_ctx,
            chat_format=chat_format,
            verbose=verbose
        )

    def chat(self, system_message, user_message):
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )
        return response["choices"][0]["message"]["content"]

class LlamaLLM(LLM):
    _llm_type: str = "deepseek-coder6.7"
    llama_wrapper: LlamaWrapper

    def __init__(self, model_path, n_gpu_layers=-1, seed=1337, n_ctx=2048, chat_format="chatml", verbose=False):
        llama_wrapper = LlamaWrapper(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            n_ctx=n_ctx,
            chat_format=chat_format,
            verbose=verbose
        )
        super().__init__(llama_wrapper = llama_wrapper)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        #system_message = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science."
        system_message = "你是一个人工智能模型，你现在要扮演一个Npc，别让别人看穿你！"
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.llama_wrapper.chat(system_message, prompt)

# 创建 LangChain 链
llama_llm = LlamaLLM(
    model_path=R"h:\LLM\TheBloke\deepseek-coder-6.7B-instruct-GGUF\deepseek-coder-6.7b-instruct.Q5_K_M.gguf"
)

prompt_template = PromptTemplate(input_variables=["user_message"], template="{user_message}")
chain = LLMChain(llm=llama_llm, prompt=prompt_template)

# 示例使用
response = chain.run(user_message="你要扮演的人物叫刘大海，是游戏里的一个初级武学训练师，想想你对第一次见到的玩家徒弟该说啥：")
print(response)