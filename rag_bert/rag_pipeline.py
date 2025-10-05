"""
RAG pipeline utilities: build_index, load_index/get_retriever, get_answers.
Matches your notebook stack: PyPDFium2 -> Recursive splitter -> OpenAIEmbeddings
(text-embedding-3-large, 3072 dims) -> Chroma -> RetrievalQA (map_reduce) with
custom prompts.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import os

from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

from prompts import QUESTION_PROMPT, COMBINE_PROMPT  # packaged import

CONFIG = {
    "embedding_model": "text-embedding-3-large",
    "embedding_dimensions": 3072,
    "persist_directory": "./chroma_store_bert_z",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "llm_model": "gpt-4o-mini",
    "llm_temperature": 0.0,
    "retrieval_k": 6
}


def build_index(pdf_path: str, persist_directory: Optional[str] = None) -> str:
    """One-time: load PDF, chunk, embed, and persist a Chroma index."""
    persist_directory = persist_directory or CONFIG["persist_directory"]
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    # 1) Load PDF
    pages = PyPDFium2Loader(pdf_path).load()

    # 2) Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
    )
    chunks = splitter.split_documents(pages)

    # 3) Embeddings
    embeddings = OpenAIEmbeddings(
        model=CONFIG["embedding_model"],
        dimensions=CONFIG["embedding_dimensions"],
    )

    # 4) Vector store
    index = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    try:
        index.persist()
    except Exception:
        pass

    return persist_directory


def load_index(persist_directory: Optional[str] = None):
    """Load a persisted Chroma index and return a retriever factory."""
    persist_directory = persist_directory or CONFIG["persist_directory"]

    embeddings = OpenAIEmbeddings(
        model=CONFIG["embedding_model"],
        dimensions=CONFIG["embedding_dimensions"],
    )
    index = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    def get_retriever(k: int = 5):
        return index.as_retriever(search_kwargs={"k": k})

    return get_retriever


def _format_sources(source_docs: List[Any]) -> Dict[str, Any]:
    sources, pages = [], []
    for d in source_docs or []:
        src = d.metadata.get("source") or d.metadata.get("filename") or "unknown"
        pg = d.metadata.get("page")
        sources.append(src)
        if isinstance(pg, int):
            pages.append(pg)
        else:
            # try cast
            try:
                pages.append(int(pg))
            except Exception:
                pass
    # unique + sorted
    return {
        "sources": list(dict.fromkeys(sources)),
        "pages": sorted(set(pages)),
    }


def get_answers(query: str, k: int = 5, retriever=None) -> Dict[str, Any]:
    """Run RetrievalQA over the loaded index and return normalized output."""
    if retriever is None:
        retriever = load_index()(k)

    llm = ChatOpenAI(model=CONFIG["llm_model"], temperature=0)

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

    result = qa.invoke({"query": query})
    answer_text = result.get("result") or result.get("answer") or ""
    src_docs = result.get("source_documents", [])

    src_info = _format_sources(src_docs)

    confidence = "unknown"
    if "i don't know" in answer_text.lower():
        confidence = "low"
    elif len(answer_text) > 60:
        confidence = "high"

    return {
        "answer": answer_text,
        "sources": src_info["sources"],
        "pages": src_info["pages"],
        "confidence": confidence,
    }
