from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os
# Load environment variables from .env file if it exists

load_dotenv()

hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
print(f"hf_token: {hf_token}")

if hf_token is None:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables.")

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct", #"TinyLlama/TinyLlama-1.1B-Chat-v1.0", --this model is not working
    task = "text-generation",
    huggingfacehub_api_token = hf_token
)
    
llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token = hf_token,
)

chat = ChatHuggingFace(llm=llm, verbose=True)
result = chat.invoke("Whats the capital of India?")
print(result.content)