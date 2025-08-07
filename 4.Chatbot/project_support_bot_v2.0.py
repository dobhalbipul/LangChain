import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

# --- Configuration Constants ---
CSV_FILE_PATH = 'tickets.csv'
ISSUE_COLUMN_NAME = 'Issue'
RESOLUTION_COLUMN_NAME = 'Resolution'
EMBEDDING_MODEL_NAME = 'mixedbread-ai/mxbai-embed-large-v1'
OLLAMA_LLM_MODEL = 'gemma3:1b'
VECTOR_DB_PATH = "faiss_index_pool" # Directory to save/load FAISS index
TOP_K_RESULTS = 3 # Number of top relevant documents to retrieve for context


# --- Streamlit UI Setup ---
st.set_page_config(page_title="ATCR Support Chatbot", page_icon="💬", layout="centered")

# --- Fixed Title/Header Section ---
# This HTML will be injected at the very top of the app's body, making it fixed.
st.markdown(
    f"""
    <div class="custom-fixed-header">
        <h1>💬 ATCR Support Chatbot</h1>
        <p>Ask me any technical issue, and I'll try to provide a solution from our knowledge base.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Helper Functions for RAG Pipeline Steps ---
def _load_documents(csv_path: str, issue_col: str, resolution_col: str, progress_bar) -> list[Document]:
    """Loads data from CSV and converts it into a list of LangChain Document objects."""
    progress_bar = st.progress(0, f"Step 1/5: Loading data from '{csv_path}'...")
    progress_bar.progress(10) # Initial progress
        
    if not os.path.exists(csv_path):
        st.sidebar.error(f"Error: The knowledge base file '{csv_path}' was not found.")
        st.stop()

    try:
        df = pd.read_csv(csv_path)
        required_columns = [issue_col, resolution_col]
        df = df[required_columns] # Filter to only required columns
        
        for col in required_columns:
            if col not in df.columns:
                st.sidebar.error(f"Error: Required column '{col}' not found in '{csv_path}'.")
                st.sidebar.error(f"Available columns are: {df.columns.tolist()}")
                st.stop()

        documents = []
        for index, row in df.iterrows():
            page_content = row[issue_col]
            # Skip rows where issue description is missing or not a string
            if not isinstance(page_content, str) or pd.isna(page_content):
                continue
            metadata = {
                "resolution": row[resolution_col],
                "original_index": index,
                "issue_description": row[issue_col]
            }
            documents.append(Document(page_content=page_content, metadata=metadata))
        
        progress_bar.progress(20) # Update progress
        st.sidebar.success(f"Loaded {len(documents)} issues from '{csv_path}'.")
        return documents
    except Exception as e:
        st.sidebar.error(f"An error occurred while loading CSV: {e}")
        st.stop()
        
        
def _load_embedding_model(model_name: str, progress_bar) -> SentenceTransformerEmbeddings:
    """Loads the SentenceTransformer embedding model."""
    progress_bar.text(f"Step 2/5: Loading embedding model: '{model_name}'...")
    progress_bar.progress(30)
    try:
        embed_model = SentenceTransformerEmbeddings(model_name=model_name)
        progress_bar.progress(40)
        st.sidebar.success("Embedding model loaded.")
        return embed_model
    except Exception as e:
        st.sidebar.error(f"Error loading embedding model '{model_name}': {e}")
        st.sidebar.error("Please check the model name and your internet connection.")
        st.stop()
        
        
def _manage_faiss_vectorstore(documents: list[Document], embed_model: SentenceTransformerEmbeddings, db_path: str, top_k: int, progress_bar):
    """
    Manages the FAISS vector store. Loads if exists, otherwise creates and saves.
    Returns a configured LangChain retriever.
    """
    vectorstore = None
    # Check if FAISS index files exist
    faiss_index_exists = (os.path.exists(db_path) and os.path.isdir(db_path) and 
                          os.path.exists(os.path.join(db_path, 'index.faiss')) and 
                          os.path.exists(os.path.join(db_path, 'index.pkl')))
    
    if faiss_index_exists:
        try:
            progress_bar.text(f"Step 3/5: Loading existing FAISS index from '{db_path}'...")
            progress_bar.progress(50)
            vectorstore = FAISS.load_local(db_path, embed_model, allow_dangerous_deserialization=True)
            st.sidebar.success("FAISS index loaded from disk.")
        except Exception as e:
            st.sidebar.warning(f"Could not load existing FAISS index: {e}. Rebuilding index...")
            vectorstore = None # Reset to trigger creation path if loading failed
    
    if vectorstore is None: # If no existing index was found or loading failed
        progress_bar.text("Step 3/5: Building new FAISS index (this may take a while)...")
        progress_bar.progress(50)
        
        with st.spinner("Generating embeddings and building new FAISS index (this may take a while)..."):
            # Ensure documents are available if rebuilding
            if not documents: # If documents were not loaded because index was *expected* to exist
                 documents = _load_documents(CSV_FILE_PATH, ISSUE_COLUMN_NAME, RESOLUTION_COLUMN_NAME, progress_bar) # Re-load them

            vectorstore = FAISS.from_documents(documents, embed_model)
            vectorstore.save_local(db_path)
        
        progress_bar.progress(60)
        st.sidebar.success(f"New FAISS index created and saved to '{db_path}'.")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    st.sidebar.success("Retriever configured.")
    return retriever

def _initialize_ollama_llm(model_name: str, progress_bar) -> Ollama:
    """Initializes the Ollama Language Model."""
    progress_bar.text(f"Step 4/5: Initializing Ollama LLM: '{model_name}'...")
    progress_bar.progress(70)
    try:
        # Use st.spinner for LLM loading too, as it can take a moment
        with st.spinner(f"Connecting to Ollama model '{model_name}'..."):
            llm = Ollama(model=model_name)
        progress_bar.progress(80)
        st.sidebar.success(f"Ollama Gemma LLM '{model_name}' initialized. Ensure `ollama serve` is running in your terminal.")
        return llm
    except Exception as e:
        st.sidebar.error(f"Error initializing Ollama Gemma: {e}")
        st.sidebar.error("Please ensure Ollama is running (`ollama serve`) and the specified model is downloaded (`ollama run gemma:2b`).")
        st.stop()
        

def _build_rag_chain(retriever, llm, progress_bar) -> RunnablePassthrough:
    """Builds the LangChain RAG pipeline."""
    progress_bar.text("Step 5/5: Building RAG chain...")
    progress_bar.progress(90)

    # Define the Prompt Template (remains the same as previous version)
    prompt_template = """
        You are a helpful IT support assistant. Your goal is to provide concise and accurate solutions to technical issues.
        You will be provided with a user's issue, problem statement and relevant context (Issue Description and Resolution) from our knowledge base.

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

    # Define a format_docs function to prepare retrieved documents for the prompt
    def format_docs(docs: list[Document]) -> str:
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
    progress_bar.progress(100) # Complete progress
    st.sidebar.success("RAG chain configured.")
    progress_bar.empty() # Remove the progress bar after completion
    return rag_chain


# --- Main RAG Pipeline Setup Function (cached) ---
@st.cache_resource(show_spinner="Setting up RAG pipeline (first run might take time)...")
def setup_full_rag_pipeline():
    """Orchestrates the setup of the entire RAG pipeline."""
    
    with st.sidebar:
        progress_text_placeholder = st.empty() # Placeholder for text
        progress_bar = st.progress(0, text="Starting setup...")
    
    documents = []
    # Check if FAISS index exists before potentially loading documents
    faiss_index_exists = (os.path.exists(VECTOR_DB_PATH) and os.path.isdir(VECTOR_DB_PATH) and 
                          os.path.exists(os.path.join(VECTOR_DB_PATH, 'index.faiss')) and 
                          os.path.exists(os.path.join(VECTOR_DB_PATH, 'index.pkl')))
    
    # Load embedding model first, as it's needed for loading or creating FAISS
    embed_model = _load_embedding_model(EMBEDDING_MODEL_NAME, progress_bar)

    if not faiss_index_exists:
        # Only load documents if we are going to create the vectorstore
        documents = _load_documents(CSV_FILE_PATH, ISSUE_COLUMN_NAME, RESOLUTION_COLUMN_NAME, progress_bar)
    

    retriever = _manage_faiss_vectorstore(documents, embed_model, VECTOR_DB_PATH, TOP_K_RESULTS, progress_bar)
    
    # Initialize Ollama LLM
    llm = _initialize_ollama_llm(OLLAMA_LLM_MODEL, progress_bar) # Pass the specific model name to the function
    
    # Build the RAG chain
    rag_chain = _build_rag_chain(retriever, llm, progress_bar)
    
    # Clear the specific progress placeholder and progress bar
    progress_text_placeholder.empty() 
    progress_bar.empty()
    st.sidebar.success("Chatbot ready!")
    
    return rag_chain


# --- Initialize RAG pipeline and chat history ---
# Call the cached main setup function
rag_chain = setup_full_rag_pipeline()

# --- Main Content (with spacer) ---
st.markdown('<div class="content-spacer">', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I assist you with your IT issue today?"})

# --- Display chat messages from history on app rerun ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])        
        
# --- User input ---
if prompt_input := st.chat_input("Ask about your ATCR issue..."):
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
with st.sidebar: # Put the clear chat button in the sidebar too
    st.markdown("---")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I assist you with your IT issue today?"})
        st.rerun()
        
st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Setup Guide:**")
st.sidebar.markdown("1. Ensure `ollama serve` is running in your terminal.")
st.sidebar.markdown("2. Download Gemma: `ollama run gemma:2b`.")
st.sidebar.markdown("3. Make sure `tickets.csv` is in the same directory as this script.")
st.sidebar.markdown("4. Run `streamlit run chatbot_app.py` in your terminal.")