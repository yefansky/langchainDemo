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
            verbose=verbose,

        )

    def chat(self, system_message, user_message):
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )
        return response["choices"][0]["message"]["content"]
    
    def stream(self, system_message, user_message) -> Iterator[str]:
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            stream=True  # 开启流式输出
        )
        
        for chunk in response:
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    yield content
                
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
        super().__init__(llama_wrapper=llama_wrapper)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        system_message = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science."
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.llama_wrapper.chat(system_message, prompt)

    def _stream(self, prompt: str, stop: Optional[List[str]] = None) -> Iterator[str]:
        system_message = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science."
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.llama_wrapper.stream(system_message, prompt)

# 创建 LangChain 链
llama_llm = LlamaLLM(
    model_path=R"h:\LLM\TheBloke\deepseek-coder-6.7B-instruct-GGUF\deepseek-coder-6.7b-instruct.Q5_K_M.gguf"
)

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate.from_template(template)
llm_chain = prompt | llama_llm

question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"

# 调用流式输出方法
for output in llama_llm._stream(question):
    print(output, end="")
