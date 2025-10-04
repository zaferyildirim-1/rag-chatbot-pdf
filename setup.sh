#!/bin/bash

# Enhanced RAG Chatbot Setup Script
# This script helps set up and run the enhanced Streamlit application

set -e  # Exit on any error

echo "🚀 Enhanced RAG Chatbot Setup"
echo "=============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if Python is installed
print_header "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Found Python $PYTHON_VERSION"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip3."
    exit 1
fi

# Create virtual environment if it doesn't exist
print_header "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
else
    print_status "Virtual environment already exists."
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install requirements
print_header "Installing Python packages..."
if [ -f "requirements.txt" ]; then
    print_status "Installing from requirements.txt..."
    pip install -r requirements.txt
else
    print_error "requirements.txt not found!"
    exit 1
fi

# Check for OpenAI API key
print_header "Checking configuration..."
if [ -z "$OPENAI_API_KEY" ]; then
    print_warning "OPENAI_API_KEY environment variable is not set."
    echo "You can set it by running:"
    echo "export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    echo "Or you can enter it in the Streamlit app interface."
else
    print_status "OpenAI API key found in environment."
fi

# Create necessary directories
print_header "Creating necessary directories..."
mkdir -p chat_sessions
mkdir -p chroma_store_bert_z
print_status "Directories created."

# Download NLTK data
print_header "Downloading NLTK data..."
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('NLTK data downloaded successfully.')
except Exception as e:
    print(f'Note: NLTK download failed: {e}')
    print('The app will still work with reduced functionality.')
"

# Check if required files exist
print_header "Verifying installation..."
required_files=(
    "streamlit_app/enhanced_app.py"
    "streamlit_app/document_processor.py"
    "streamlit_app/chat_persistence.py"
    "streamlit_app/evaluation_metrics.py"
    "rag_bert/__init__.py"
    "rag_bert/rag_pipeline.py"
    "rag_bert/prompts.py"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    print_error "Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    exit 1
else
    print_status "All required files found."
fi

# Test import of main modules
print_status "Testing module imports..."
python3 -c "
import sys
import os
sys.path.insert(0, '.')

try:
    import streamlit
    print('✓ Streamlit imported successfully')
except ImportError as e:
    print(f'✗ Streamlit import failed: {e}')
    sys.exit(1)

try:
    from langchain_community.document_loaders import PyPDFium2Loader
    print('✓ LangChain community imported successfully')
except ImportError as e:
    print(f'✗ LangChain community import failed: {e}')
    sys.exit(1)

try:
    from langchain_openai import OpenAIEmbeddings
    print('✓ LangChain OpenAI imported successfully')
except ImportError as e:
    print(f'✗ LangChain OpenAI import failed: {e}')
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
    print('✓ Data analysis libraries imported successfully')
except ImportError as e:
    print(f'✗ Data analysis libraries import failed: {e}')
    sys.exit(1)

print('All core modules imported successfully!')
"

print_status "Installation verification complete!"

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "To run the enhanced RAG chatbot:"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Set your OpenAI API key (if not already set):"
echo "   export OPENAI_API_KEY='your-api-key-here'"
echo ""
echo "3. Run the enhanced Streamlit app:"
echo "   streamlit run streamlit_app/enhanced_app.py"
echo ""
echo "Or run the original simple app:"
echo "   streamlit run streamlit_app/app.py"
echo ""
echo "The app will open in your browser at http://localhost:8501"
echo ""
echo "📋 Features available:"
echo "• Enhanced UI with custom styling"
echo "• Advanced document processing"
echo "• Chat session persistence"
echo "• Performance analytics"
echo "• Multiple chunking strategies"
echo "• Export capabilities"
echo ""
echo "📁 File locations:"
echo "• Chat sessions: ./chat_sessions/"
echo "• Vector store: ./chroma_store_bert_z/"
echo "• Sample document: ./bert_article.pdf"
echo ""
echo "For help and documentation, see README_enhanced.md"
echo ""

# Ask if user wants to start the app immediately
read -p "Would you like to start the enhanced app now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Starting the enhanced Streamlit app..."
    streamlit run streamlit_app/enhanced_app.py
fi