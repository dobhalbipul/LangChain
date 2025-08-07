import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

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
st.title("Chatbot with LangChain and Google Generative AI")
input_text = st.text_input("Enter your question:", key="txt_input")
question = input_text.strip() if input_text else None

# create gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.2,
    max_output_tokens=1024,
    top_p=0.95,
    top_k=40,
)

# output parser
output_parser = StrOutputParser()

# langchain chain
chain = prompt_template | model | output_parser

# run the chain and display the result
if question:
    st.write(chain.invoke({"question": question}))
