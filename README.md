
# Langchain

## Project Overview
Langchain is a collection of demos and utilities for working with LLMs, chat models, and embedding models using Python. The project is organized into several modules:

- **1.LLMs/**: Demos and notebooks for working with various LLMs.
- **2.ChatModels/**: Scripts and notebooks for different chat model providers (Anthropic, Google, OpenAI, HuggingFace).
- **3.EmbeddingModels/**: Embedding demos, custom RAG pools, and FAISS index files.
- **4.Chatbot/**: Chatbot applications, including support bots and local Ollama integration.

## Directory Structure
```
1.LLMs/
	1_llm_demo.py
	1.langchain_demo.ipynb
2.ChatModels/
	2_chatmodel_anthropic.py
	2_chatmodel_google.ipynb
	2_chatmodel_google.py
	2_chatmodel_openai.py
	4_chatmodel_huggingface.py
3.EmbeddingModels/
	amazon_river_info.txt
	custom_emmed_rag_pool.ipynb
	embedding_google.ipynb
	embedding_google2.ipynb
	embedding_rag.ipynb
	huggingface_rag_pool.ipynb
	ollama_pool.ipynb
	tickets.csv
	faiss_index_pool/
		index.faiss
		index.pkl
4.Chatbot/
	app.py
	local_ollama.py
	project_support_bot_v2.0.py
	project_support_bot.py
	tickets_template.csv
	tickets.csv
	faiss_index_pool/
		index.faiss
		index.pkl
```

## Getting Started
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Explore the modules and notebooks for demos and usage examples.

## Requirements
See `requirements.txt` for all dependencies.

## License
MIT
