# --- robust package import shim (must be first) ---
import os, sys
from pathlib import Path

APP_FILE = Path(__file__).resolve()
REPO_ROOT = APP_FILE.parents[1]          # repo root expected: .../rag-chatbot-pdf
PKG_DIR = REPO_ROOT / "rag_bert"         # must exist: .../rag_bert

# Ensure repo root is on sys.path so "rag_bert" is importable as a package
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Hard checks so we fail with a clear message if layout is wrong
assert PKG_DIR.exists(), f"Expected package folder not found: {PKG_DIR}"
assert (PKG_DIR / "__init__.py").exists(), f"Missing __init__.py in {PKG_DIR} (create an empty file)."
# --------------------------------------------------


import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

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

from rag_pipeline import CONFIG as PIPE_CFG  # noqa
# Correct package import
from rag_bert.rag_pipeline import load_index as load_persisted_index
from rag_bert.prompts import QUESTION_PROMPT, COMBINE_PROMPT
st.set_page_config(page_title="RAG PDF Chatbot", layout="wide")
st.title("📄 RAG PDF Chatbot — upload a PDF and chat")

with st.sidebar:
    st.header("Session settings")
    k = st.slider("Retriever k", 1, 10, 5)
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    use_persisted = st.toggle("Use persisted index (server-side)", value=False)
    st.caption(f"Persist dir: `{PIPE_CFG['persist_directory']}`")

    if api_key:
        # Ensure all LangChain OpenAI clients see the key
        os.environ["OPENAI_API_KEY"] = api_key

# Chat history
if "msgs" not in st.session_state:
    st.session_state.msgs = []

uploaded = st.file_uploader("Upload a PDF to chat with", type=["pdf"])
user_q = st.text_input("Ask a question about the PDF:")

def _build_vectorstore_from_pdf(file) -> Chroma:
    # Save to a temp file for PyPDFium2Loader
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    pages = PyPDFium2Loader(tmp_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=PIPE_CFG["chunk_size"], chunk_overlap=PIPE_CFG["chunk_overlap"]
    )
    chunks = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(
        model=PIPE_CFG["embedding_model"],
        dimensions=PIPE_CFG["embedding_dimensions"],
    )
    # In-memory Chroma (non-persisted) for user uploads
    vs = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vs

def _make_qa_chain(vs: Chroma, k: int) -> RetrievalQA:
    retriever = vs.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=PIPE_CFG["llm_model"], temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        chain_type_kwargs={
            "question_prompt": QUESTION_PROMPT,
            "combine_prompt": COMBINE_PROMPT,
        },
        return_source_documents=False,
    )
    return qa

# Build or load the vector store
vectorstore = None
status = st.empty()

if use_persisted:
    if not api_key:
        st.info("Enter your OpenAI API key to use the persisted index.")
    else:
        try:
            get_retriever = load_persisted_index(PIPE_CFG["persist_directory"])
            # wrap a tiny adapter that RetrievalQA expects (has .as_retriever)
            class _Adapter:
                def __init__(self, rgetter): self._rg = rgetter
                def as_retriever(self, search_kwargs=None): return self._rg(search_kwargs.get("k", 5))
            vectorstore = _Adapter(get_retriever)
            status.success("Loaded persisted index.")
        except Exception as e:
            status.error(f"Failed to load persisted index: {e}")

elif uploaded and api_key:
    with st.spinner("Processing PDF (chunk → embed → index)…"):
        try:
            vectorstore = _build_vectorstore_from_pdf(uploaded)
            status.success("PDF processed. Ready to chat.")
        except Exception as e:
            status.error(f"Failed to process PDF: {e}")

# Ask/answer
if user_q:
    if not api_key:
        st.warning("Please enter your OpenAI API key.")
    elif vectorstore is None:
        st.info("Please upload and process a PDF, or load a persisted index.")
    else:
        qa = _make_qa_chain(vectorstore, k=k)
        try:
            ans = qa.invoke({"query": user_q})
            text = ans.get("result") if isinstance(ans, dict) else str(ans)
        except Exception:
            # Fallback to .run for older LC behavior
            text = qa.run(user_q)
        st.session_state.msgs.append(("user", user_q))
        st.session_state.msgs.append(("assistant", text))

# Render chat
st.subheader("Chat history")
for role, content in st.session_state.msgs:
    with st.chat_message(role):
        st.markdown(content)

# Footer info
with st.expander("Session info"):
    st.write({"Retriever k": k,
              "Using persisted index": bool(use_persisted and vectorstore is not None)})
