import streamlit as st
import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader
from langchain_community.document_loaders import Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import ollama
import tempfile
from pathlib import Path
import shutil
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DEFAULT_MODEL_NAME = "llama3.2"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"
SUPPORTED_EXTENSIONS = ['.pdf', '.txt', '.docx', '.csv']

# Page configuration
st.set_page_config(
    page_title="Smart Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .css-1d391kg {
        padding: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .stAlert {
        animation: slideIn 0.3s;
    }
    @keyframes slideIn {
        from { transform: translateX(-100%); }
        to { transform: translateX(0); }
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = set()
    if "current_chain" not in st.session_state:
        st.session_state.current_chain = None
    if "total_chunks" not in st.session_state:
        st.session_state.total_chunks = 0

def get_file_loader(file_path, file_extension):
    """Return appropriate loader based on file type."""
    if file_extension == '.pdf':
        return UnstructuredPDFLoader(file_path)
    elif file_extension == '.txt':
        return TextLoader(file_path, encoding='utf-8')
    elif file_extension == '.docx':
        return Docx2txtLoader(file_path)
    elif file_extension == '.csv':
        return CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def process_uploaded_files(uploaded_files, progress_bar, status_text):
    """Process and load uploaded files."""
    all_documents = []
    
    for idx, uploaded_file in enumerate(uploaded_files):
        if uploaded_file.name in st.session_state.processed_files:
            continue
            
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # Load document
            file_extension = Path(uploaded_file.name).suffix.lower()
            loader = get_file_loader(tmp_path, file_extension)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name
                doc.metadata["file_type"] = file_extension
            
            all_documents.extend(documents)
            st.session_state.processed_files.add(uploaded_file.name)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
        
        # Update progress
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    return all_documents

def create_vector_store(documents, embedding_model, vector_db_type):
    """Create vector store from documents."""
    embedding = OllamaEmbeddings(model=embedding_model)
    
    if vector_db_type == "Chroma":
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding,
            collection_name="rag_collection"
        )
    else:  # FAISS
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embedding
        )
    
    return vector_store

def create_chain(retriever, llm, system_prompt, chunk_size):
    """Create the processing chain with custom prompt."""
    template = f"""You are a helpful AI assistant that answers questions based on the provided context.
    
Instructions:
- Answer based ONLY on the following context
- If you don't know the answer, say "I cannot find this information in the provided documents"
- Be concise but thorough
- Use a friendly and professional tone

System instruction: {system_prompt}

Context:
{{context}}

Question: {{question}}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def main():
    initialize_session_state()
    
    # Header
    st.markdown("<h1 style='text-align: center;'>📚 Smart Document Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Your AI-powered document analysis companion</p>", unsafe_allow_html=True)
    st.divider()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
        st.markdown("## ⚙️ Configuration")
        
        # Model selection
        st.markdown("### 🤖 Model Settings")
        model_name = st.selectbox(
            "LLM Model",
            options=["llama3.2", "llama3.1", "mistral", "phi3"],
            index=0,
            help="Select the language model to use"
        )
        
        embedding_model = st.selectbox(
            "Embedding Model",
            options=["nomic-embed-text", "llama3.2", "mxbai-embed-large"],
            index=0,
            help="Model for document embeddings"
        )
        
        # Vector database options
        st.markdown("### 🗄️ Vector Database")
        vector_db_type = st.radio(
            "Database Type",
            options=["Chroma", "FAISS"],
            help="Chroma is persistent, FAISS is faster for in-memory operations"
        )
        
        # Chunking parameters
        st.markdown("### ✂️ Chunking Settings")
        chunk_size = st.slider(
            "Chunk Size",
            min_value=500,
            max_value=2000,
            value=1200,
            step=100,
            help="Size of text chunks for processing"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Overlap between chunks to maintain context"
        )
        
        # Retrieval settings
        st.markdown("### 🔍 Retrieval Settings")
        k_retrieval = st.slider(
            "Number of chunks to retrieve",
            min_value=1,
            max_value=10,
            value=4,
            help="How many relevant chunks to fetch"
        )
        
        search_type = st.selectbox(
            "Search Type",
            options=["similarity", "mmr"],
            help="MMR provides more diverse results"
        )
        
        # System prompt customization
        st.markdown("### 💬 Assistant Personality")
        system_prompt = st.text_area(
            "Custom Instructions",
            value="Be helpful, accurate, and concise in your responses.",
            height=100,
            help="Custom instructions for the AI assistant"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content area - two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("## 📤 Document Upload")
        st.markdown("Upload one or more documents to analyze")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'docx', 'csv'],
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Documents", use_container_width=True):
                with st.spinner("Processing documents..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process documents
                    documents = process_uploaded_files(uploaded_files, progress_bar, status_text)
                    
                    if documents:
                        # Split documents
                        status_text.text("Splitting documents into chunks...")
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        chunks = text_splitter.split_documents(documents)
                        st.session_state.total_chunks = len(chunks)
                        
                        # Create vector store
                        status_text.text("Creating vector database...")
                        st.session_state.vector_store = create_vector_store(
                            chunks, embedding_model, vector_db_type
                        )
                        
                        # Create retriever and chain
                        retriever = st.session_state.vector_store.as_retriever(
                            search_type=search_type,
                            search_kwargs={"k": k_retrieval}
                        )
                        
                        llm = ChatOllama(model=model_name)
                        st.session_state.current_chain = create_chain(
                            retriever, llm, system_prompt, chunk_size
                        )
                        
                        status_text.text("✅ Ready!")
                        time.sleep(1)
                        status_text.empty()
                        progress_bar.empty()
                        
                        st.success(f"Successfully processed {len(uploaded_files)} file(s) into {len(chunks)} chunks!")
                        
                        # Display metrics
                        st.markdown("### 📊 Processing Metrics")
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Files Processed", len(uploaded_files))
                        with col_b:
                            st.metric("Total Chunks", len(chunks))
                        with col_c:
                            st.metric("Chunk Size", f"{chunk_size} chars")
                    else:
                        st.error("No documents were processed successfully.")
        
        # Display processed files
        if st.session_state.processed_files:
            st.markdown("### 📄 Processed Files")
            for file in st.session_state.processed_files:
                st.markdown(f"- ✅ {file}")
            
            if st.button("🗑️ Clear All Documents", use_container_width=True):
                st.session_state.processed_files.clear()
                st.session_state.vector_store = None
                st.session_state.current_chain = None
                st.session_state.messages = []
                st.session_state.total_chunks = 0
                st.rerun()
    
    with col2:
        st.markdown("## 💬 Chat Interface")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            if st.session_state.current_chain is not None:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.current_chain.invoke(prompt)
                            st.markdown(response)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_msg = f"Error generating response: {str(e)}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            else:
                with st.chat_message("assistant"):
                    st.warning("Please upload and process documents first before asking questions.")
    
    # Footer
    st.divider()
    st.markdown(
        "<p style='text-align: center; font-size: 0.8rem;'>Powered by LangChain, Ollama, and Streamlit</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()