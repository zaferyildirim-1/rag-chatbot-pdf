# RAG PDF Chatbot - Academic Paper Assistant

A sophisticated Retrieval-Augmented Generation (RAG) chatbot designed specifically for academic papers and research documents. Built with Streamlit, LangChain, and OpenAI APIs.

## 🚀 Features

### Core Functionality
- **PDF Upload & Processing**: Upload academic papers and research documents
- **Advanced RAG Pipeline**: Intelligent document chunking, embedding, and retrieval
- **Interactive Chat Interface**: Natural language queries about your documents
- **Multiple Chunking Strategies**: Recursive, section-based, and semantic chunking
- **Persistent Vector Storage**: Save and reuse document indexes

### Enhanced UI/UX
- **Modern Streamlit Interface**: Clean, responsive design with custom CSS
- **Progress Tracking**: Real-time processing indicators
- **Session Management**: Save and load chat sessions
- **Quick Actions**: Pre-built question templates
- **Document Analysis**: Metadata extraction and statistics

### Analytics & Evaluation
- **Performance Metrics**: Response time, accuracy, and quality scores
- **Session Analytics**: Question complexity, response coherence analysis
- **Source Attribution**: Track and display source relevance
- **Export Capabilities**: Download chat sessions and metrics

## 📋 Requirements

### Core Dependencies
```
streamlit==1.50.0
openai==1.55.3
pypdfium2==4.30.0
langchain==0.1.17
langchain-openai==0.1.6
langchain-community==0.0.38
langchain-chroma==0.1.0
chromadb==0.4.24
```

### Enhanced Features
```
pandas==2.0.3
numpy==1.24.3
plotly==5.15.0
seaborn==0.12.2
matplotlib==3.7.2
sentence-transformers==2.2.2
scikit-learn==1.3.0
nltk==3.8.1
streamlit-chat==0.1.1
streamlit-extras==0.3.0
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd rag-chatbot-pdf-main
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   ```

## 🚀 Usage

### Basic Usage
1. **Start the application**:
   ```bash
   streamlit run streamlit_app/enhanced_app.py
   ```

2. **Upload a PDF**: Use the file uploader to select your academic paper

3. **Configure settings**: Adjust retrieval parameters, chunking strategy, and model settings

4. **Start chatting**: Ask questions about your document using natural language

### Advanced Features

#### Multiple Chunking Strategies
- **Recursive**: Standard text splitting with overlap
- **Section-based**: Splits by document sections and headers
- **Semantic**: Groups semantically similar content together

#### Session Management
- **Save sessions**: Preserve chat history with metadata
- **Load sessions**: Resume previous conversations
- **Export sessions**: Download as text or JSON

#### Performance Analytics
- **Real-time metrics**: Response time, similarity scores
- **Session analytics**: Question complexity, response quality
- **Source tracking**: Relevance and diversity of retrieved content

## 📁 Project Structure

```
rag-chatbot-pdf-main/
├── rag_bert/                          # Core RAG pipeline
│   ├── __init__.py
│   ├── rag_pipeline.py                # Main RAG functionality
│   ├── prompts.py                     # LLM prompts
│   └── evaluate.py                    # Evaluation utilities
├── streamlit_app/                     # Streamlit applications
│   ├── app.py                         # Original simple app
│   ├── enhanced_app.py                # Main enhanced application
│   ├── document_processor.py          # Advanced document processing
│   ├── chat_persistence.py            # Session management
│   └── evaluation_metrics.py          # Performance analytics
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
├── bert_article.pdf                  # Sample document
└── *.ipynb                           # Jupyter notebooks
```

## 🎯 Key Components

### 1. Enhanced Streamlit App (`enhanced_app.py`)
- Modern UI with custom CSS styling
- Real-time processing indicators
- Advanced configuration options
- Interactive chat interface
- Performance dashboard

### 2. Document Processor (`document_processor.py`)
- Multiple chunking strategies
- Metadata extraction (title, authors, abstract)
- Document analysis and statistics
- Chunk distribution visualization

### 3. Chat Persistence (`chat_persistence.py`)
- Session saving and loading
- JSON export functionality
- Auto-save capabilities
- Session management UI

### 4. Evaluation Metrics (`evaluation_metrics.py`)
- Response quality assessment
- Semantic similarity analysis
- Source relevance scoring
- Session-level analytics

## 🔧 Configuration

### Model Settings
- **LLM Model**: GPT-4o-mini (configurable)
- **Embedding Model**: text-embedding-3-large
- **Embedding Dimensions**: 3072
- **Temperature**: 0.1 (adjustable)

### Chunking Parameters
- **Chunk Size**: 1000 tokens (adjustable)
- **Chunk Overlap**: 200 tokens (adjustable)
- **Retrieval K**: 5 chunks (adjustable)

### Storage
- **Vector Database**: ChromaDB
- **Persistence**: Local filesystem
- **Session Storage**: JSON files

## 📊 Analytics & Metrics

The system provides comprehensive analytics including:

### Response Metrics
- Semantic similarity with question
- Response coherence
- Processing time
- Source relevance

### Session Analytics
- Question complexity analysis
- Response quality trends
- Source diversity
- Performance over time

### Export Options
- JSON metrics export
- Text session export
- Performance reports

## 🤝 Usage Examples

### Academic Research
```python
# Example questions for research papers:
"What is the main contribution of this paper?"
"What methodology does the paper use?"
"What are the key results and findings?"
"How does this work compare to previous research?"
"What are the limitations mentioned?"
"What future work is suggested?"
```

### Document Analysis
```python
# The system can help with:
- Summarizing complex academic content
- Extracting key methodologies
- Identifying research gaps
- Understanding technical concepts
- Comparing different approaches
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **OpenAI API Issues**: Verify your API key is set correctly
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

3. **Memory Issues**: For large documents, reduce chunk size or limit retrieval count

4. **NLTK Data**: The system will auto-download required NLTK data

### Performance Tips

1. **Optimize Chunk Size**: Smaller chunks for specific queries, larger for context
2. **Adjust Retrieval Count**: More sources for complex questions
3. **Use Persistence**: Save processed documents to avoid reprocessing
4. **Monitor Metrics**: Use analytics to optimize performance

## 🚀 Future Enhancements

- [ ] Multi-document chat capability
- [ ] Advanced citation and reference extraction
- [ ] Integration with academic databases
- [ ] Support for more document formats
- [ ] Advanced evaluation metrics
- [ ] User authentication and sharing
- [ ] API endpoint for integration

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Built with ❤️ for the academic research community**