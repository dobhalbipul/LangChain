import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
#load environment variables from .env file
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["LANGIAN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions from the user."),
    ("user", "{question}"),
])

# streamlit app
st.set_page_config(page_title="Chatbot", page_icon=":robot:")
st.title("Chatbot with LangChain with Ollama model")
input_text = st.text_input("Enter your question:", key="txt_input")
question = input_text.strip() if input_text else None

# create ollama model, make sure you have ollama installed and running
# you can install ollama from https://ollama.com/download
# to check the ollama model in your machine, run the command "ollama list" in command line
# to download the model, go to https://github.com/ollama/ollama and run the command "ollama run gemma3:1b"
model = Ollama(model="gemma3:1b")

# output parser
output_parser = StrOutputParser()

# langchain chain
chain = prompt_template | model | output_parser

# run the chain and display the result
if question:
    st.write(chain.invoke({"question": question}))
