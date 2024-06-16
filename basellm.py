from llama_cpp import Llama

model_path=R"h:\LLM\TheBloke\deepseek-coder-6.7B-instruct-GGUF\deepseek-coder-6.7b-instruct.Q5_K_M.gguf"

llm = Llama(
      model_path=model_path,
      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      seed=1337, # Uncomment to set a specific seed
      n_ctx=2048, # Uncomment to increase the context window
      chat_format="chatml",
      verbose=False
)

response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are an AI programming assistant, utilizing the Deepseek Coder model, developed by Deepseek Company, and you only answer questions related to computer science.",
        },
        {"role": "user", "content": "write a yolov1 algorithm"},
    ]
)

print(response["choices"][0]["message"]["content"])