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
    st.error(f"Expected package folder not found: {PKG_DIR}")
    st.stop()
if not (PKG_DIR / "__init__.py").exists():
    st.warning(f"Missing __init__.py in {PKG_DIR} - creating it")
    (PKG_DIR / "__init__.py").touch()
# --------------------------------------------------

import os
import sys
import tempfile
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import streamlit as st

# Core imports with error handling
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    st.warning("Pandas not available - some features will be limited")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    st.warning("Plotly not available - using basic charts")

# LangChain bits matching your notebook
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Make package imports work both locally and on Streamlit Cloud
ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / "rag_bert"
if str(PKG_DIR) not in sys.path:
    sys.path.append(str(PKG_DIR))

try:
    from rag_pipeline import CONFIG as PIPE_CFG  # noqa
    from rag_bert.rag_pipeline import load_index as load_persisted_index
    from rag_bert.prompts import QUESTION_PROMPT, COMBINE_PROMPT
except ImportError as e:
    st.error(f"Could not import rag_bert modules: {e}")
    # Fallback configuration
    PIPE_CFG = {
        "embedding_model": "text-embedding-3-large",
        "embedding_dimensions": 3072,
        "persist_directory": "./chroma_store_bert_z",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "llm_model": "gpt-4o-mini",
    }
    
    # Simple fallback prompts
    from langchain.prompts import PromptTemplate
    QUESTION_PROMPT = PromptTemplate(
        template="Use the following pieces of context to answer the question at the end.\n\n{context}\n\nQuestion: {question}\nAnswer:",
        input_variables=["context", "question"]
    )
    COMBINE_PROMPT = PromptTemplate(
        template="Given the following extracted parts of a long document and a question, create a final answer.\n\n{summaries}\n\nQuestion: {question}\nAnswer:",
        input_variables=["summaries", "question"]
    )

# Page configuration
st.set_page_config(
    page_title="RAG PDF Chatbot", 
    layout="wide",
    page_icon="📄",
    initial_sidebar_state="expanded"
)

# Simplified CSS for cloud compatibility
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1e88e5;
    }
    .assistant-message {
    background-color: transparent; /* or #ffffff or #f8f9fa */
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
    border-left: 1px solid #e0e0e0; /* subtle */
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📄 Academic Paper RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown("### Upload academic papers and chat with intelligent AI assistance")

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key", 
        type="password", 
        help="Enter your OpenAI API key to enable the chatbot"
    )
    
    # Model configuration
    st.subheader("Model Settings")
    k = st.slider(
        "Number of retrieved chunks", 
        min_value=1, 
        max_value=15, 
        value=5,
        help="Number of document chunks to retrieve for each query"
    )
    
    temperature = st.slider(
        "Response creativity", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1,
        step=0.1,
        help="Higher values make responses more creative but less focused"
    )
    
    # Advanced options
    with st.expander("🔧 Advanced Options"):
        chunk_size = st.number_input(
            "Chunk size", 
            min_value=200, 
            max_value=2000, 
            value=PIPE_CFG["chunk_size"],
            help="Size of text chunks for processing"
        )
        chunk_overlap = st.number_input(
            "Chunk overlap", 
            min_value=0, 
            max_value=500, 
            value=PIPE_CFG["chunk_overlap"],
            help="Overlap between consecutive chunks"
        )
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.msgs = []
        st.session_state.chat_stats = {"questions": 0, "processing_times": []}
        st.rerun()

# Initialize session state
if "msgs" not in st.session_state:
    st.session_state.msgs = []

if "chat_stats" not in st.session_state:
    st.session_state.chat_stats = {
        "questions": 0,
        "processing_times": [],
        "session_start": datetime.now()
    }

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Set API key in environment
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # File upload section
    st.subheader("📤 Document Upload")
    uploaded = st.file_uploader(
        "Upload a PDF to chat with", 
        type=["pdf"],
        help="Upload academic papers, research documents, or any PDF file"
    )
    
    # Document processing functions
    def _build_vectorstore_from_pdf(file) -> tuple:
        """Build vector store from uploaded PDF with progress tracking"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Save to temp file
            status_text.text("💾 Saving uploaded file...")
            progress_bar.progress(10)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            
            # Load PDF
            status_text.text("📖 Loading PDF pages...")
            progress_bar.progress(30)
            pages = PyPDFium2Loader(tmp_path).load()
            
            # Split documents
            status_text.text("✂️ Splitting document into chunks...")
            progress_bar.progress(50)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_documents(pages)
            
            # Create embeddings
            status_text.text("🧠 Creating embeddings...")
            progress_bar.progress(70)
            embeddings = OpenAIEmbeddings(
                model=PIPE_CFG["embedding_model"],
                dimensions=PIPE_CFG["embedding_dimensions"],
            )
            
            # Build vector store
            status_text.text("🔍 Building search index...")
            progress_bar.progress(90)
            vs = Chroma.from_documents(documents=chunks, embedding=embeddings)
            
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            return vs, len(pages), len(chunks)
            
        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")
            return None, 0, 0
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def _make_qa_chain(vs: Chroma, k: int, temp: float) -> RetrievalQA:
        """Create QA chain with specified parameters"""
        retriever = vs.as_retriever(search_kwargs={"k": k})
        llm = ChatOpenAI(model=PIPE_CFG["llm_model"], temperature=temp)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="map_reduce",
            chain_type_kwargs={
                "question_prompt": QUESTION_PROMPT,
                "combine_prompt": COMBINE_PROMPT,
            },
            return_source_documents=True,
        )
        return qa
    
    # Process uploaded file
    if uploaded and api_key:
        if st.button("🚀 Process Document", use_container_width=True):
            with st.spinner("Processing your document..."):
                result = _build_vectorstore_from_pdf(uploaded)
                if result[0] is not None:
                    st.session_state.vectorstore = result[0]
                    st.session_state.doc_stats = {
                        "pages": result[1],
                        "chunks": result[2],
                        "filename": uploaded.name
                    }
                    st.success(f"✅ Successfully processed {uploaded.name}!")
                    st.balloons()

# Display chat history
if st.session_state.msgs:
    st.subheader("� Conversation")
    
    # Create a container for the chat messages
    chat_container = st.container()
    
    with chat_container:
        # Display messages in chronological order (most recent at bottom)
        for i, msg in enumerate(st.session_state.msgs):
            if msg[0] == "user":
                st.markdown(f"""
                <div class="user-message">
                    <strong>🙋 You:</strong> {msg[1]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    <strong>🤖 Assistant:</strong> {msg[1]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources if available
                if len(msg) > 2 and msg[2]:
                    with st.expander(f"📚 Sources used", expanded=False):
                        for j, source in enumerate(msg[2][:3]):  # Show top 3 sources
                            st.markdown(f"**Source {j+1}:**")
                            st.text(source.page_content[:300] + "..." if len(source.page_content) > 300 else source.page_content)
                
                # Add a small separator after each assistant response
                st.markdown("---")
    
    # Show a simple hint for continuing the conversation
    if len(st.session_state.msgs) > 0:
        st.info("💡 **Continue the conversation below!**")

else:
    # Show helpful prompts when no conversation has started
    st.info("👋 **Welcome!** Upload a PDF document and start asking questions. Here are some ideas:")
    
    col_idea1, col_idea2 = st.columns(2)
    with col_idea1:
        st.markdown("""
        **📚 Research Questions:**
        - What is this paper about?
        - What problem does it solve?
        - What are the key findings?
        - How was the study conducted?
        """)
    
    with col_idea2:
        st.markdown("""
        **🔍 Deep Dive Questions:**
        - Can you explain [specific concept]?
        - What are the limitations?
        - How does this compare to other work?
        - What are the implications?
        """)

# Chat Interface - Now at the bottom for better flow
st.markdown("---")

# Initialize chat input state
if "chat_input_key" not in st.session_state:
    st.session_state.chat_input_key = 0

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
    
    # Create a form for better chat flow
    with st.form(key=f"chat_form_{st.session_state.chat_input_key}", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1])
        
        with col_input:
            user_q = st.text_input(
                "Type your question here:", 
                value=follow_up_value,  # Pre-fill with follow-up question if available
                placeholder="Ask me anything about this document...",
                key=f"question_input_{st.session_state.chat_input_key}",
                label_visibility="collapsed"
            )
        
        with col_send:
            send_button = st.form_submit_button("� Send", use_container_width=True)

    # Process question when form is submitted
    if send_button and user_q:
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
                
                # Auto-scroll to bottom by rerunning
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")

elif not api_key:
    st.markdown("---")
    st.info("🔑 **Enter your OpenAI API key in the sidebar to start chatting**")
elif st.session_state.vectorstore is None:
    st.markdown("---")
    st.info("📄 **Upload and process a PDF document to start chatting**")

# Simple visual indicator for active chat
if st.session_state.vectorstore is not None and api_key and len(st.session_state.msgs) > 0:
    st.markdown("*Ready for your next question...*")

# Sidebar statistics
with col2:
    st.subheader("📊 Session Statistics")
    
    # Document stats
    if hasattr(st.session_state, 'doc_stats'):
        with st.container():
            st.metric("📄 Document", st.session_state.doc_stats["filename"])
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("📖 Pages", st.session_state.doc_stats["pages"])
            with col_b:
                st.metric("📝 Chunks", st.session_state.doc_stats["chunks"])
    
    # Chat stats
    stats = st.session_state.chat_stats
    with st.container():
        st.metric("❓ Questions Asked", stats["questions"])
        
        if stats["processing_times"]:
            avg_time = sum(stats["processing_times"]) / len(stats["processing_times"])
            st.metric("⚡ Avg Response Time", f"{avg_time:.2f}s")
            
            # Simple response time chart (fallback if plotly not available)
            if HAS_PLOTLY and len(stats["processing_times"]) > 1:
                fig = px.line(
                    x=list(range(1, len(stats["processing_times"]) + 1)),
                    y=stats["processing_times"],
                    title="Response Times"
                )
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
            elif len(stats["processing_times"]) > 1:
                # Fallback to simple line chart
                if HAS_PANDAS:
                    chart_data = pd.DataFrame({
                        "Question": list(range(1, len(stats["processing_times"]) + 1)),
                        "Time (seconds)": stats["processing_times"]
                    })
                    st.line_chart(chart_data.set_index("Question"))

# Footer with system info
with st.expander("🔧 System Information"):
    col_x, col_y = st.columns(2)
    
    with col_x:
        st.write({
            "Retriever k": k,
            "Temperature": temperature,
            "Chunk size": chunk_size,
            "Chunk overlap": chunk_overlap,
        })
    
    with col_y:
        st.write({
            "LLM Model": PIPE_CFG["llm_model"],
            "Embedding Model": PIPE_CFG["embedding_model"],
            "Session duration": str(datetime.now() - stats["session_start"]).split('.')[0],
            "Has Pandas": HAS_PANDAS,
            "Has Plotly": HAS_PLOTLY
        })
