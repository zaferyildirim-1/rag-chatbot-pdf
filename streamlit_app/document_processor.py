"""
Enhanced document processing utilities for the RAG chatbot.
Provides advanced chunking strategies, metadata extraction, and document analysis.
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import tempfile

from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st


class EnhancedDocumentProcessor:
    """Enhanced document processor with multiple chunking strategies and metadata extraction."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def extract_document_metadata(self, pages: List[Document]) -> Dict:
        """Extract metadata from document pages."""
        try:
            full_text = "\n\n".join([page.page_content for page in pages])
            
            # Extract potential title (first significant line)
            lines = full_text.split('\n')
            title = None
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if len(line) > 10 and not line.isdigit() and len(line) < 200:
                    title = line
                    break
            
            # Extract authors (look for common patterns)
            authors = []
            author_patterns = [
                r'Authors?:\s*([^\n]+)',
                r'By\s+([^\n]+)',
                r'^([A-Z][a-z]+\s+[A-Z][a-z]+(?:,\s*[A-Z][a-z]+\s+[A-Z][a-z]+)*)',
            ]
            
            for pattern in author_patterns:
                matches = re.findall(pattern, full_text[:2000], re.MULTILINE)
                if matches:
                    authors.extend(matches)
                    break
            
            # Extract abstract
            abstract = None
            abstract_patterns = [
                r'Abstract[:\-\s]*([^\.]+(?:\.[^\.]+){2,5}\.)',
                r'ABSTRACT[:\-\s]*([^\.]+(?:\.[^\.]+){2,5}\.)',
            ]
            
            for pattern in abstract_patterns:
                match = re.search(pattern, full_text[:3000], re.DOTALL | re.IGNORECASE)
                if match:
                    abstract = match.group(1).strip()
                    break
            
            # Count words and estimate reading time
            word_count = len(full_text.split())
            reading_time = max(1, word_count // 200)  # ~200 words per minute
            
            return {
                "title": title or "Unknown Title",
                "authors": authors,
                "abstract": abstract,
                "pages": len(pages),
                "word_count": word_count,
                "reading_time_minutes": reading_time,
                "character_count": len(full_text)
            }
            
        except Exception as e:
            st.warning(f"Could not extract all metadata: {e}")
            return {
                "title": "Unknown Title",
                "authors": [],
                "abstract": None,
                "pages": len(pages),
                "word_count": 0,
                "reading_time_minutes": 0,
                "character_count": 0
            }
    
    def chunk_by_sections(self, pages: List[Document]) -> List[Document]:
        """Chunk document by sections (headers, paragraphs)."""
        chunks = []
        
        for page in pages:
            content = page.page_content
            
            # Split by common section indicators
            section_patterns = [
                r'\n\d+\.?\s+[A-Za-z]',  # Numbered sections (1. Introduction)
                r'\n[A-Z][A-Z\s]{3,}\n',  # ALL CAPS headers
                r'\n[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\n',  # Title Case headers
                r'\n\n\s*\n',  # Multiple line breaks
            ]
            
            # Try to split by sections first
            sections = [content]
            for pattern in section_patterns:
                new_sections = []
                for sec in sections:
                    new_sections.extend(re.split(pattern, sec))
                sections = new_sections
                if len(sections) > 2:  # If we found good splits, use them
                    break
            
            # Process each section
            for i, section in enumerate(sections):
                section = section.strip()
                if len(section) < 50:  # Skip very short sections
                    continue
                
                # If section is still too long, split it further
                if len(section) > self.chunk_size * 1.5:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap
                    )
                    sub_chunks = splitter.split_text(section)
                    
                    for j, sub_chunk in enumerate(sub_chunks):
                        chunk_doc = Document(
                            page_content=sub_chunk,
                            metadata={
                                **page.metadata,
                                "chunk_type": "section_split",
                                "section_index": i,
                                "sub_chunk_index": j,
                                "total_sections": len(sections)
                            }
                        )
                        chunks.append(chunk_doc)
                else:
                    chunk_doc = Document(
                        page_content=section,
                        metadata={
                            **page.metadata,
                            "chunk_type": "section",
                            "section_index": i,
                            "total_sections": len(sections)
                        }
                    )
                    chunks.append(chunk_doc)
        
        return chunks
    
    def chunk_by_semantic_similarity(self, pages: List[Document]) -> List[Document]:
        """Chunk document trying to keep semantically similar content together."""
        # This is a simplified version - in a full implementation,
        # you'd use sentence embeddings to group similar sentences
        
        chunks = []
        
        for page in pages:
            content = page.page_content
            
            # Split into sentences
            sentences = re.split(r'[.!?]+\s+', content)
            
            current_chunk = ""
            current_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                    # Create chunk from accumulated sentences
                    chunk_doc = Document(
                        page_content=current_chunk.strip(),
                        metadata={
                            **page.metadata,
                            "chunk_type": "semantic",
                            "sentence_count": len(current_sentences)
                        }
                    )
                    chunks.append(chunk_doc)
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0 and current_sentences:
                        overlap_sentences = current_sentences[-2:]  # Keep last 2 sentences
                        current_chunk = " ".join(overlap_sentences) + " " + sentence
                        current_sentences = overlap_sentences + [sentence]
                    else:
                        current_chunk = sentence
                        current_sentences = [sentence]
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
                    current_sentences.append(sentence)
            
            # Add final chunk if it has content
            if current_chunk.strip():
                chunk_doc = Document(
                    page_content=current_chunk.strip(),
                    metadata={
                        **page.metadata,
                        "chunk_type": "semantic",
                        "sentence_count": len(current_sentences)
                    }
                )
                chunks.append(chunk_doc)
        
        return chunks
    
    def process_document(self, file, chunking_strategy: str = "recursive") -> Tuple[List[Document], Dict]:
        """
        Process uploaded document with specified chunking strategy.
        
        Args:
            file: Uploaded file object
            chunking_strategy: "recursive", "sections", or "semantic"
            
        Returns:
            Tuple of (chunks, metadata)
        """
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        try:
            # Load PDF pages
            loader = PyPDFium2Loader(tmp_path)
            pages = loader.load()
            
            # Extract metadata
            metadata = self.extract_document_metadata(pages)
            metadata["filename"] = file.name
            metadata["chunking_strategy"] = chunking_strategy
            
            # Apply chunking strategy
            if chunking_strategy == "sections":
                chunks = self.chunk_by_sections(pages)
            elif chunking_strategy == "semantic":
                chunks = self.chunk_by_semantic_similarity(pages)
            else:  # Default recursive
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                chunks = splitter.split_documents(pages)
                
                # Add metadata to chunks
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_type": "recursive",
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    })
            
            metadata["total_chunks"] = len(chunks)
            metadata["avg_chunk_size"] = sum(len(chunk.page_content) for chunk in chunks) // len(chunks)
            
            return chunks, metadata
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
    
    def analyze_chunk_distribution(self, chunks: List[Document]) -> Dict:
        """Analyze the distribution of chunk sizes and types."""
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        chunk_types = [chunk.metadata.get("chunk_type", "unknown") for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
            "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            "avg_chunk_size": sum(chunk_sizes) // len(chunk_sizes) if chunk_sizes else 0,
            "chunk_type_distribution": {
                chunk_type: chunk_types.count(chunk_type) 
                for chunk_type in set(chunk_types)
            },
            "size_distribution": {
                "under_500": sum(1 for size in chunk_sizes if size < 500),
                "500_1000": sum(1 for size in chunk_sizes if 500 <= size < 1000),
                "1000_1500": sum(1 for size in chunk_sizes if 1000 <= size < 1500),
                "over_1500": sum(1 for size in chunk_sizes if size >= 1500),
            }
        }


def display_document_analysis(metadata: Dict, chunk_analysis: Dict):
    """Display document and chunk analysis in Streamlit."""
    
    st.subheader("📊 Document Analysis")
    
    # Basic document info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📄 Pages", metadata.get("pages", 0))
        st.metric("📝 Words", f"{metadata.get('word_count', 0):,}")
    
    with col2:
        st.metric("🔤 Characters", f"{metadata.get('character_count', 0):,}")
        st.metric("⏱️ Est. Reading Time", f"{metadata.get('reading_time_minutes', 0)} min")
    
    with col3:
        st.metric("📋 Total Chunks", chunk_analysis.get("total_chunks", 0))
        st.metric("📊 Avg Chunk Size", chunk_analysis.get("avg_chunk_size", 0))
    
    # Document metadata
    with st.expander("📋 Document Details"):
        if metadata.get("title"):
            st.write(f"**Title:** {metadata['title']}")
        
        if metadata.get("authors"):
            st.write(f"**Authors:** {', '.join(metadata['authors'][:3])}")
        
        if metadata.get("abstract"):
            st.write("**Abstract:**")
            st.write(metadata["abstract"][:500] + "..." if len(metadata["abstract"]) > 500 else metadata["abstract"])
    
    # Chunk distribution visualization
    if chunk_analysis.get("size_distribution"):
        st.subheader("📈 Chunk Size Distribution")
        
        size_dist = chunk_analysis["size_distribution"]
        sizes = list(size_dist.keys())
        counts = list(size_dist.values())
        
        # Create a simple bar chart
        chart_data = {"Size Range": sizes, "Count": counts}
        st.bar_chart(chart_data, x="Size Range", y="Count")