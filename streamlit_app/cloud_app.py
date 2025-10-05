# --- robust package import shim (must be first) ---
import os, sys
from pathlib import Path

APP_FILE = Path(__file__).resolve()
REPO_ROOT = APP_FILE.parents[1]          # repo root expected: .../rag-chatbot-pdf
PKG_DIR = REPO_ROOT / "rag_bert"         # must exist: .../rag_bert

# Ensure repo root is on sys.path so "rag_bert" is importable as a package
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Soft checks for cloud deployment
if not PKG_DIR.exists():
    # Will show error after streamlit import
    PKG_DIR_EXISTS = False
else:
    PKG_DIR_EXISTS = True
    if not (PKG_DIR / "__init__.py").exists():
        (PKG_DIR / "__init__.py").touch()
# --------------------------------------------------

import os
import sys
import tempfile
import json
import time
from pathlib import Path
from datetime import datetime

import streamlit as st

# Check if package directory exists (delayed from earlier)
if not PKG_DIR_EXISTS:
    st.error(f"Expected package folder not found: {PKG_DIR}")
    st.stop()

# Plotting libraries (optional for cloud deployment)
try:
    import pandas as pd
except ImportError:
    pd = None
    st.warning("pandas not available - some features disabled")

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = go = None
    st.warning("plotly not available - chart features disabled")

# LangChain imports
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Try to import from our package
try:
    from rag_pipeline import CONFIG as PIPE_CFG  # noqa
except ImportError:
    st.warning("rag_pipeline not found - using defaults")
    PIPE_CFG = {
        "embedding_model": "text-embedding-3-large", 
        "embedding_dimensions": 3072,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "llm_model": "gpt-4o-mini",
        "llm_temperature": 0.0,
        "retrieval_k": 6
    }

# Import prompts from the package if available
try:
    from langchain.prompts import PromptTemplate
    QUESTION_PROMPT = PromptTemplate(
        template="""Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}
        Answer:""",
        input_variables=["context", "question"]
    )
    
    COMBINE_PROMPT = PromptTemplate(
        template="""Given the following extracted parts of a long document and a question, create a final answer. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Question: {question}
        =========
        {summaries}
        =========
        Answer:""",
        input_variables=["summaries", "question"]
    )
except ImportError:
    QUESTION_PROMPT = COMBINE_PROMPT = None

# Page config
st.set_page_config(
    page_title="RAG PDF Chatbot", 
    layout="wide",
    page_icon="📄",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1e88e5;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #8e24aa;
    }
    .source-card {
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.2rem 0;
        border: 1px solid #dee2e6;
        font-size: 0.85rem;
    }
    .stTextInput > div > div > input {
        font-size: 1rem;
    }
    .processing-info {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def _make_qa_chain(vectorstore, k=6, temp=0.0):
    """Create a QA chain with the vectorstore."""
    llm = ChatOpenAI(
        model=PIPE_CFG["llm_model"], 
        temperature=temp
    )
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    chain_kwargs = {
        "chain_type": "map_reduce",
        "return_source_documents": True,
    }
    
    if QUESTION_PROMPT and COMBINE_PROMPT:
        chain_kwargs["chain_type_kwargs"] = {
            "question_prompt": QUESTION_PROMPT,
            "combine_prompt": COMBINE_PROMPT
        }
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        **chain_kwargs
    )

def load_and_split_pdf(uploaded_file, chunk_size=512, chunk_overlap=64):
    """Load and split a PDF file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file.flush()
        
        loader = PyPDFium2Loader(tmp_file.name)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        split_docs = text_splitter.split_documents(documents)
        
        # Clean up temp file
        os.unlink(tmp_file.name)
        
        return split_docs

def create_vectorstore(documents, api_key):
    """Create a vectorstore from documents."""
    embeddings = OpenAIEmbeddings(
        model=PIPE_CFG["embedding_model"],
        openai_api_key=api_key
    )
    
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=None  # In-memory for cloud deployment
    )
    
    return vectorstore

# Initialize session state
if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0
if "chat_stats" not in st.session_state:
    st.session_state.chat_stats = {
        "questions": 0,
        "processing_times": [],
        "document_name": None,
        "session_start": datetime.now()
    }
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# Main header
st.markdown('<h1 class="main-header">📄 RAG PDF Chatbot</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key", 
        type="password",
        help="Enter your OpenAI API key to enable AI features",
        placeholder="sk-..."
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.success("✅ API key configured")
    else:
        st.warning("⚠️ API key required")
    
    st.divider()
    
    # Advanced settings
    with st.expander("⚙️ Advanced Settings"):
        k = st.slider("Number of retrieved chunks", 1, 10, PIPE_CFG["retrieval_k"])
        temperature = st.slider("LLM Temperature", 0.0, 1.0, PIPE_CFG["llm_temperature"], 0.1)
        chunk_size = st.slider("Chunk Size", 256, 1024, PIPE_CFG["chunk_size"], 64)
        chunk_overlap = st.slider("Chunk Overlap", 32, 256, PIPE_CFG["chunk_overlap"], 16)
    
    st.divider()
    
    # Statistics
    if st.session_state.chat_stats["questions"] > 0:
        st.header("📊 Session Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Questions", st.session_state.chat_stats["questions"])
        
        with col2:
            avg_time = sum(st.session_state.chat_stats["processing_times"]) / len(st.session_state.chat_stats["processing_times"])
            st.metric("Avg Response Time", f"{avg_time:.1f}s")
        
        if st.session_state.chat_stats["document_name"]:
            st.text(f"Document: {st.session_state.chat_stats['document_name']}")
        
        # Processing time chart
        if pd and px and len(st.session_state.chat_stats["processing_times"]) > 1:
            times_df = pd.DataFrame({
                'Question': range(1, len(st.session_state.chat_stats["processing_times"]) + 1),
                'Time (s)': st.session_state.chat_stats["processing_times"]
            })
            fig = px.line(times_df, x='Question', y='Time (s)', title='Response Times')
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.msgs = []
        st.session_state.chat_stats = {
            "questions": 0,
            "processing_times": [],
            "document_name": st.session_state.chat_stats.get("document_name"),
            "session_start": datetime.now()
        }
        st.rerun()

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Choose a PDF file", 
        type="pdf",
        help="Upload a PDF document to chat with"
    )
    
    if uploaded_file and api_key:
        # Check if we need to process a new file
        if (st.session_state.vectorstore is None or 
            st.session_state.chat_stats["document_name"] != uploaded_file.name):
            
            with st.spinner("📖 Processing PDF... This may take a moment."):
                try:
                    # Load and split the PDF
                    documents = load_and_split_pdf(uploaded_file, chunk_size, chunk_overlap)
                    
                    # Create vectorstore
                    st.session_state.vectorstore = create_vectorstore(documents, api_key)
                    st.session_state.chat_stats["document_name"] = uploaded_file.name
                    
                    st.success(f"✅ Successfully processed {len(documents)} chunks from {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
                    st.session_state.vectorstore = None

with col2:
    if st.session_state.vectorstore is not None:
        st.success("📄 Document Ready")
        st.info(f"File: {st.session_state.chat_stats['document_name']}")
    else:
        st.info("📄 No document loaded")

# Chat interface
if st.session_state.msgs:
    st.markdown("### 💬 Conversation")
    
    for i, (role, content, *sources) in enumerate(st.session_state.msgs):
        with st.container():
            if role == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong> {content}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 Assistant:</strong> {content}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources if available
                if sources and sources[0]:
                    with st.expander(f"📚 Sources ({len(sources[0])} chunks)", expanded=False):
                        for j, doc in enumerate(sources[0][:5]):  # Limit to 5 sources
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Source {j+1}:</strong><br>
                                {doc.page_content[:300]}{'...' if len(doc.page_content) > 300 else ''}
                            </div>
                            """, unsafe_allow_html=True)

# Initialize processing state
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# Handle follow-up questions
follow_up_value = ""
if "follow_up_question" in st.session_state:
    follow_up_value = st.session_state.follow_up_question
    del st.session_state.follow_up_question

# Show input only if document is processed and API key is provided
if st.session_state.vectorstore is not None and api_key:
    # Chat input at the bottom with clean styling
    st.markdown("---")
    st.markdown("### 💬 Ask your next question:")
    
    # Show processing status if currently processing
    if st.session_state.is_processing:
        st.markdown("""
        <div class="processing-info">
            🤔 <strong>Processing your question...</strong><br>
            Please wait for the response before asking another question.
        </div>
        """, unsafe_allow_html=True)
    
    # Create a form for better chat flow (disabled during processing)
    with st.form(key=f"chat_form_{st.session_state.chat_input_key}", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1])
        
        with col_input:
            user_q = st.text_input(
                "Type your question here:", 
                value=follow_up_value if not st.session_state.is_processing else "",  # Clear if processing
                placeholder="Processing... Please wait..." if st.session_state.is_processing else "Ask me anything about this document...",
                key=f"question_input_{st.session_state.chat_input_key}",
                label_visibility="collapsed",
                disabled=st.session_state.is_processing
            )
        
        with col_send:
            send_button = st.form_submit_button(
                "⏳ Wait..." if st.session_state.is_processing else "📤 Send", 
                use_container_width=True,
                disabled=st.session_state.is_processing
            )

    # Process question when form is submitted
    if send_button and user_q and not st.session_state.is_processing:
        # Set processing state
        st.session_state.is_processing = True
        
        # Increment the key to reset the form
        st.session_state.chat_input_key += 1
        
        with st.spinner("🤔 Thinking..."):
            start_time = time.time()
            
            try:
                # Add conversation context for follow-up questions
                contextual_query = user_q
                if len(st.session_state.msgs) >= 2:  # If there's previous conversation
                    last_q = st.session_state.msgs[-2][1] if st.session_state.msgs[-2][0] == "user" else ""
                    last_a = st.session_state.msgs[-1][1] if st.session_state.msgs[-1][0] == "assistant" else ""
                    
                    # Check if this might be a follow-up question
                    follow_up_indicators = ["this", "that", "it", "explain", "clarify", "more", "previous", "above"]
                    if any(indicator in user_q.lower() for indicator in follow_up_indicators):
                        contextual_query = f"Previous question: {last_q}\nPrevious answer: {last_a[:200]}...\n\nNew question: {user_q}"
                
                qa = _make_qa_chain(st.session_state.vectorstore, k=k, temp=temperature)
                result = qa.invoke({"query": contextual_query})
                
                if isinstance(result, dict):
                    answer = result.get("result", str(result))
                    sources = result.get("source_documents", [])
                else:
                    answer = str(result)
                    sources = []
                
                processing_time = time.time() - start_time
                
                # Update session stats
                st.session_state.chat_stats["questions"] += 1
                st.session_state.chat_stats["processing_times"].append(processing_time)
                
                # Add to chat history
                st.session_state.msgs.append(("user", user_q))
                st.session_state.msgs.append(("assistant", answer, sources))
                
                # Clear processing state
                st.session_state.is_processing = False
                
                # Auto-scroll to bottom by rerunning
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                # Clear processing state on error
                st.session_state.is_processing = False

elif not api_key:
    st.markdown("---")
    st.info("🔑 **Enter your OpenAI API key in the sidebar to start chatting**")
elif st.session_state.vectorstore is None:
    st.markdown("---")
    st.info("📄 **Upload and process a PDF document to start chatting**")

# Footer
with st.container():
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Made with ❤️ using Streamlit and LangChain | "
        f"Session started: {st.session_state.chat_stats['session_start'].strftime('%H:%M:%S')}"
        "</div>", 
        unsafe_allow_html=True
    )