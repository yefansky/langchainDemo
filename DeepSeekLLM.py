from llama_cpp import Llama
from langchain_core.language_models.llms import LLM
from typing import Any, Dict, Iterator, List, Mapping, Optional
from pydantic import BaseModel

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

class DeepSeekLLM(LLM):
    llama_wrapper: Optional[LlamaWrapper] = None
    system_prompt: str = "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science."
    _llm_type: str = "deepseek-coder6.7"

    def __init__(self, model_path, n_gpu_layers=-1, seed=1337, n_ctx=2048, chat_format="chatml", verbose=False, system_prompt=None, **kwargs):
        super().__init__(**kwargs)
        self.llama_wrapper = LlamaWrapper(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            n_ctx=n_ctx,
            chat_format=chat_format,
            verbose=verbose
        )
        if system_prompt:
            self.system_prompt = system_prompt

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.llama_wrapper.chat(self.system_prompt, prompt)

    def _stream(self, prompt: str, stop: Optional[List[str]] = None) -> Iterator[str]:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.llama_wrapper.stream(self.system_prompt, prompt)

    @property
    def llm_type(self) -> str:
        return self._llm_type
