# 🚀 Quick Start Guide - Enhanced RAG Chatbot

## ⚡ Fastest Way to Get Started

### Option 1: Automated Setup (Recommended)
```bash
cd /Users/esmasert/Downloads/rag-chatbot-pdf-main
./setup.sh
```

### Option 2: Manual Setup
```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key
export OPENAI_API_KEY="your-openai-api-key-here"

# 4. Run the app
streamlit run streamlit_app/enhanced_app.py
```

## 🎯 First Steps After Installation

1. **Upload a PDF**: Start with the included `bert_article.pdf` or upload your own
2. **Configure Settings**: Adjust chunk size, retrieval count, and temperature in the sidebar
3. **Ask Questions**: Try these sample questions:
   - "What is the main contribution of this paper?"
   - "What methodology does this paper use?"
   - "What are the key results?"

## 🔧 Configuration Tips

### For Academic Papers:
- **Chunk Size**: 1000-1500 tokens for detailed papers
- **Retrieval K**: 5-8 chunks for comprehensive answers
- **Temperature**: 0.1-0.3 for factual responses

### For Quick Queries:
- **Chunk Size**: 500-800 tokens for focused answers
- **Retrieval K**: 3-5 chunks for concise responses
- **Temperature**: 0.0-0.2 for precise answers

## 📊 New Features You'll Love

### Enhanced UI
- **Custom Styling**: Professional dark/light theme
- **Progress Bars**: Real-time processing feedback
- **Quick Actions**: Pre-built question templates
- **Statistics Dashboard**: Live performance metrics

### Advanced Processing
- **Multiple Chunking**: Choose between recursive, section-based, or semantic
- **Document Analysis**: Automatic metadata extraction
- **Source Attribution**: See exactly which parts of the document were used

### Session Management
- **Save/Load**: Preserve your conversations
- **Export Options**: Download as text or JSON
- **Auto-save**: Never lose your progress

### Analytics
- **Response Quality**: AI-powered quality scoring
- **Performance Metrics**: Speed, relevance, coherence
- **Session Analytics**: Track improvements over time

## 🔍 Troubleshooting

### Common Issues:

**App won't start?**
```bash
# Check Python version (need 3.8+)
python3 --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**OpenAI errors?**
```bash
# Verify API key
echo $OPENAI_API_KEY

# Test API connection
python3 -c "import openai; print('API key configured')"
```

**Import errors?**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Install missing packages
pip install streamlit langchain-openai langchain-community
```

## 🎨 Customization

### Change App Theme
Edit `enhanced_app.py`, modify the CSS in the `st.markdown()` section around line 50.

### Add New Question Templates
Edit the quick action buttons at the bottom of `enhanced_app.py`.

### Modify Chunking Strategy
Adjust parameters in `document_processor.py` or add new strategies.

## 📱 Mobile Usage

The enhanced app is mobile-responsive! Use it on tablets and phones for reading and querying documents on the go.

## 🔗 Integration

### Use as a Library
```python
from rag_bert.rag_pipeline import build_index, load_index
from streamlit_app.evaluation_metrics import RAGEvaluator

# Build your own custom applications
```

### API Integration
The core RAG functionality can be wrapped in FastAPI for programmatic access.

## 🎯 Pro Tips

1. **Upload Multiple Documents**: Process different papers in separate sessions
2. **Use Persistence**: Save processed documents to avoid reprocessing
3. **Monitor Analytics**: Use metrics to understand what works best
4. **Export Sessions**: Keep records of important research insights
5. **Experiment with Settings**: Different papers may need different configurations

## 🚀 What's Next?

After you're comfortable with the basic features, explore:
- Advanced chunking strategies for complex documents
- Session analytics to improve your research workflow  
- Export capabilities for sharing insights
- Performance tuning for optimal response quality

---

**Happy researching! 📚✨**

Need help? Check `README_enhanced.md` for detailed documentation or create an issue on GitHub.