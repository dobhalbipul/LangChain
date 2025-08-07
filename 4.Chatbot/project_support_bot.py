import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

# --- Configuration ---
CSV_FILE_PATH = 'tickets.csv'
ISSUE_COLUMN_NAME = 'Issue'
RESOLUTION_COLUMN_NAME = 'Resolution'
EMBEDDING_MODEL_NAME = 'mixedbread-ai/mxbai-embed-large-v1'
VECTOR_DB_PATH = "faiss_index_pool" # Directory to save/load FAISS index
TOP_K_RESULTS = 2 # Number of top relevant documents to retrieve for context

# --- Streamlit UI Setup ---
st.set_page_config(page_title="ATCR Support Chatbot", page_icon="💬", layout="centered")

st.title("💬 ATCR Support Chatbot")
st.write("Ask me any technical issue, and I'll try to provide a solution from our knowledge base.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Progress...**")
# --- Function to load data and set up RAG components (cached for efficiency) ---
@st.cache_resource
def setup_rag_pipeline():
    """
    Loads data, initializes the embedding model, creates/loads the FAISS vector store,
    and sets up the RAG LangChain. This function runs only once per app deployment.
    """
    # st.info("Initializing chatbot components... This might take a moment on first run.")
    st.sidebar.markdown("Initializing chatbot components... This might take a moment on first run.")
    # 1. Load Data
    if not os.path.exists(CSV_FILE_PATH):
        st.error(f"Error: The knowledge base file '{CSV_FILE_PATH}' was not found.")
        st.stop() # Stop the app if the CSV is missing

    try:
        df = pd.read_csv(CSV_FILE_PATH)

        required_columns = [ISSUE_COLUMN_NAME, RESOLUTION_COLUMN_NAME]
        df = df[required_columns] # Filter to only required columns
        
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Error: Required column '{col}' not found in '{CSV_FILE_PATH}'.")
                st.error(f"Available columns are: {df.columns.tolist()}")
                st.stop()

        documents = []
        for index, row in df.iterrows():
            page_content = row[ISSUE_COLUMN_NAME]
            
            # Skip rows where issue description is missing or not a string
            if not isinstance(page_content, str) or pd.isna(page_content):
                continue
            metadata = {
                "resolution": row[RESOLUTION_COLUMN_NAME],
                "original_index": index,
                "issue_description": row[ISSUE_COLUMN_NAME]
            }
            documents.append(Document(page_content=page_content, metadata=metadata))
        
        st.sidebar.markdown(f"Loaded {len(documents)} issues from '{CSV_FILE_PATH}'.")

    except Exception as e:
        st.error(f"An error occurred while loading CSV: {e}")
        st.stop()

    # 2. Load Embedding Model
    try:
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        embed_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        st.sidebar.markdown("Embedding model loaded.")
    except Exception as e:
        st.error(f"Error loading embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        st.error("Please check the model name and your internet connection.")
        st.stop()

    # 3. Create/Load Vector Store (FAISS)
    try:
        if os.path.exists(VECTOR_DB_PATH) and os.path.isdir(VECTOR_DB_PATH):
            vectorstore = FAISS.load_local(VECTOR_DB_PATH, embed_model, allow_dangerous_deserialization=True)
            st.sidebar.markdown("FAISS index loaded from disk.")
        else:
            st.info("Creating new FAISS index and generating embeddings (this may take a while)...")
            vectorstore = FAISS.from_documents(documents, embed_model)
            vectorstore.save_local(VECTOR_DB_PATH)
            st.sidebar.markdown(f"FAISS index created and saved to '{VECTOR_DB_PATH}'.")

        retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
        st.sidebar.markdown("Retriever configured.")
    except Exception as e:
        st.error(f"Error with FAISS vector store: {e}")
        st.stop()

    # 4. Initialize Ollama LLM
    try:
        llm = Ollama(model="gemma3:1b")
        st.sidebar.markdown("Ollama Gemma LLM initialized. Ensure `ollama serve` is running in your terminal.")
    except Exception as e:
        st.error(f"Error initializing Ollama Gemma: {e}")
        st.stop()

    # 5. Define the Prompt Template
    prompt_template = """
        You are a helpful IT support assistant. Your goal is to provide concise and accurate solutions to technical issues.
        You will be provided with a user's technical issue and relevant context (Issue Description and Resolution) from our knowledge base.

        If the provided context contains relevant information to address the user's input, synthesize the best possible solution from the 'Resolution' in the context.
        Prioritize the provided resolution in your answer.


        If the provided context contains multiple resolutions, combine them and provide bulleted list of probable solutions.
        Keep the response as professional support assistant.
        Correct the spelling, grammer in the solution and paraphrase always, if needed.
        If the context does NOT contain any information directly relevant to the user's input, or if you cannot form a solution from the provided context,
        then you MUST respond with: "Please raise a pool ticket for this issue." 
        User's Input: {question}

        Context:
        {context}

        Solution:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # 6. Set up the RAG Chain
    def format_docs(docs):
        formatted_context = ""
        for i, doc in enumerate(docs):
            formatted_context += f"Issue {i+1}: {doc.metadata['issue_description']}\n"
            formatted_context += f"Resolution {i+1}: {doc.metadata['resolution']}\n\n"
        return formatted_context.strip()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # st.success("RAG chain configured.")

    return rag_chain


# --- Initialize RAG pipeline and chat history ---
rag_chain = setup_rag_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I assist you with your ATCR issue today?"})

# --- Display chat messages from history on app rerun ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User input ---
if prompt_input := st.chat_input("Ask about your issue..."):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke(prompt_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred while generating a response: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "I apologize, but I encountered an error. Please try again or raise a pool ticket."})

# Optional: Clear chat history button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I assist you with your IT issue today?"})
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Setup Guide:**")
st.sidebar.markdown("1. Ensure `ollama serve` is running in your terminal.")
st.sidebar.markdown("2. Download Gemma: `ollama run gemma:2b`.")
st.sidebar.markdown("3. Make sure `tickets.csv` is in the same directory as this script.")
st.sidebar.markdown("4. Run `streamlit run chatbot_app.py` in your terminal.")