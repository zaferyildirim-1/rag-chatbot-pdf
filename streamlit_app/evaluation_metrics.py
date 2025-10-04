"""
Evaluation and metrics module for the RAG chatbot.
Provides various metrics to assess the quality of responses and retrieval.
"""

import re
import time
from typing import List, Dict, Tuple, Optional
from collections import Counter
import json

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data (with error handling)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass


class RAGEvaluator:
    """Evaluates RAG system performance with various metrics."""
    
    def __init__(self):
        self.stop_words = set()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stop words if NLTK data unavailable
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def calculate_response_metrics(self, 
                                 question: str, 
                                 response: str, 
                                 sources: List = None,
                                 processing_time: float = None) -> Dict:
        """Calculate comprehensive metrics for a single response."""
        
        metrics = {}
        
        # Basic metrics
        metrics.update(self._basic_text_metrics(question, response))
        
        # Semantic metrics
        metrics.update(self._semantic_metrics(question, response))
        
        # Source relevance metrics
        if sources:
            metrics.update(self._source_relevance_metrics(question, response, sources))
        
        # Performance metrics
        if processing_time:
            metrics['processing_time'] = processing_time
            metrics['response_speed'] = len(response.split()) / max(processing_time, 0.1)  # words per second
        
        return metrics
    
    def _basic_text_metrics(self, question: str, response: str) -> Dict:
        """Calculate basic text metrics."""
        
        # Response length metrics
        response_words = response.split()
        response_sentences = sent_tokenize(response) if response else []
        
        # Question-response relationship
        question_words = set(word.lower().strip('.,!?') for word in question.split())
        response_words_set = set(word.lower().strip('.,!?') for word in response_words)
        
        # Remove stop words for better analysis
        question_content_words = question_words - self.stop_words
        response_content_words = response_words_set - self.stop_words
        
        word_overlap = len(question_content_words & response_content_words)
        question_coverage = word_overlap / max(len(question_content_words), 1)
        
        return {
            'response_length_words': len(response_words),
            'response_length_chars': len(response),
            'response_sentences': len(response_sentences),
            'avg_sentence_length': len(response_words) / max(len(response_sentences), 1),
            'question_word_coverage': question_coverage,
            'unique_words_ratio': len(response_words_set) / max(len(response_words), 1)
        }
    
    def _semantic_metrics(self, question: str, response: str) -> Dict:
        """Calculate semantic similarity metrics."""
        
        try:
            # Use TF-IDF for semantic similarity
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            
            # Combine question and response for vectorization
            texts = [question, response]
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Calculate response coherence (average similarity between sentences)
            sentences = sent_tokenize(response)
            coherence_score = 0.0
            
            if len(sentences) > 1:
                sentence_vectors = vectorizer.transform(sentences)
                coherence_scores = []
                
                for i in range(len(sentences) - 1):
                    sent_sim = cosine_similarity(
                        sentence_vectors[i:i+1], 
                        sentence_vectors[i+1:i+2]
                    )[0][0]
                    coherence_scores.append(sent_sim)
                
                coherence_score = np.mean(coherence_scores) if coherence_scores else 0.0
            
            return {
                'semantic_similarity': similarity,
                'response_coherence': coherence_score,
                'tfidf_score': tfidf_matrix[1].sum()
            }
            
        except Exception as e:
            return {
                'semantic_similarity': 0.0,
                'response_coherence': 0.0,
                'tfidf_score': 0.0
            }
    
    def _source_relevance_metrics(self, question: str, response: str, sources: List) -> Dict:
        """Calculate metrics related to source relevance and usage."""
        
        if not sources:
            return {'source_relevance': 0.0, 'source_coverage': 0.0}
        
        try:
            # Combine source content
            source_texts = [source.page_content for source in sources if hasattr(source, 'page_content')]
            combined_sources = ' '.join(source_texts)
            
            # Calculate question-source relevance
            vectorizer = TfidfVectorizer(stop_words='english')
            vectors = vectorizer.fit_transform([question, combined_sources])
            question_source_sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            # Calculate response-source relevance
            if response:
                response_vectors = vectorizer.transform([response])
                response_source_sim = cosine_similarity(response_vectors, vectors[1:2])[0][0]
            else:
                response_source_sim = 0.0
            
            # Calculate source diversity (how different the sources are from each other)
            source_diversity = 0.0
            if len(source_texts) > 1:
                source_vectors = vectorizer.transform(source_texts)
                similarities = []
                for i in range(len(source_texts)):
                    for j in range(i+1, len(source_texts)):
                        sim = cosine_similarity(source_vectors[i:i+1], source_vectors[j:j+1])[0][0]
                        similarities.append(sim)
                source_diversity = 1.0 - np.mean(similarities) if similarities else 0.0
            
            return {
                'question_source_relevance': question_source_sim,
                'response_source_relevance': response_source_sim,
                'source_diversity': source_diversity,
                'num_sources_used': len(sources)
            }
            
        except Exception:
            return {
                'question_source_relevance': 0.0,
                'response_source_relevance': 0.0,
                'source_diversity': 0.0,
                'num_sources_used': len(sources)
            }
    
    def calculate_session_metrics(self, messages: List[Tuple]) -> Dict:
        """Calculate metrics for an entire chat session."""
        
        if not messages:
            return {}
        
        user_questions = []
        assistant_responses = []
        all_sources = []
        
        # Extract questions and responses
        for msg in messages:
            if len(msg) >= 2:
                if msg[0] == 'user':
                    user_questions.append(msg[1])
                elif msg[0] == 'assistant':
                    assistant_responses.append(msg[1])
                    if len(msg) > 2 and msg[2]:
                        all_sources.extend(msg[2])
        
        if not user_questions or not assistant_responses:
            return {}
        
        # Calculate aggregate metrics
        session_metrics = {
            'total_questions': len(user_questions),
            'total_responses': len(assistant_responses),
            'avg_question_length': np.mean([len(q.split()) for q in user_questions]),
            'avg_response_length': np.mean([len(r.split()) for r in assistant_responses]),
            'total_sources_retrieved': len(all_sources),
            'avg_sources_per_response': len(all_sources) / max(len(assistant_responses), 1)
        }
        
        # Question complexity analysis
        question_complexities = [self._estimate_question_complexity(q) for q in user_questions]
        session_metrics['avg_question_complexity'] = np.mean(question_complexities)
        
        # Response quality distribution
        response_qualities = []
        for i, response in enumerate(assistant_responses):
            if i < len(user_questions):
                quality = self._estimate_response_quality(user_questions[i], response)
                response_qualities.append(quality)
        
        if response_qualities:
            session_metrics['avg_response_quality'] = np.mean(response_qualities)
            session_metrics['response_quality_std'] = np.std(response_qualities)
        
        return session_metrics
    
    def _estimate_question_complexity(self, question: str) -> float:
        """Estimate question complexity based on various factors."""
        
        complexity_score = 0.0
        
        # Length factor
        word_count = len(question.split())
        if word_count > 10:
            complexity_score += 0.3
        if word_count > 20:
            complexity_score += 0.2
        
        # Question words that indicate complexity
        complex_words = ['analyze', 'compare', 'evaluate', 'explain', 'describe', 'discuss', 'why', 'how']
        for word in complex_words:
            if word in question.lower():
                complexity_score += 0.1
        
        # Multiple questions in one
        if '?' in question:
            question_marks = question.count('?')
            if question_marks > 1:
                complexity_score += 0.2 * (question_marks - 1)
        
        # Technical terms (rough heuristic)
        technical_indicators = len(re.findall(r'\b[A-Z]{2,}\b', question))  # Acronyms
        complexity_score += technical_indicators * 0.1
        
        return min(complexity_score, 1.0)  # Cap at 1.0
    
    def _estimate_response_quality(self, question: str, response: str) -> float:
        """Estimate response quality based on various factors."""
        
        if not response or not question:
            return 0.0
        
        quality_score = 0.0
        
        # Length appropriateness (not too short, not too long)
        response_length = len(response.split())
        if 20 <= response_length <= 200:
            quality_score += 0.3
        elif 10 <= response_length < 20 or 200 < response_length <= 300:
            quality_score += 0.2
        elif response_length < 10:
            quality_score += 0.1
        
        # Semantic similarity with question
        try:
            vectorizer = TfidfVectorizer(stop_words='english')
            vectors = vectorizer.fit_transform([question, response])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            quality_score += similarity * 0.4
        except:
            pass
        
        # Structure quality (sentences, punctuation)
        sentences = sent_tokenize(response)
        if len(sentences) >= 2:
            quality_score += 0.2
        
        # Avoid repetition
        words = response.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        quality_score += unique_ratio * 0.1
        
        return min(quality_score, 1.0)  # Cap at 1.0


def display_evaluation_dashboard(evaluator: RAGEvaluator, 
                               chat_history: List[Tuple],
                               current_metrics: Dict = None):
    """Display evaluation dashboard in Streamlit."""
    
    st.subheader("📊 Performance Evaluation")
    
    if not chat_history:
        st.info("No chat history to evaluate yet.")
        return
    
    # Calculate session metrics
    session_metrics = evaluator.calculate_session_metrics(chat_history)
    
    if not session_metrics:
        st.info("Not enough data for evaluation.")
        return
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Questions Asked", int(session_metrics.get('total_questions', 0)))
        st.metric("💬 Responses Given", int(session_metrics.get('total_responses', 0)))
    
    with col2:
        avg_q_len = session_metrics.get('avg_question_length', 0)
        st.metric("📏 Avg Question Length", f"{avg_q_len:.1f} words")
        avg_r_len = session_metrics.get('avg_response_length', 0)
        st.metric("📄 Avg Response Length", f"{avg_r_len:.1f} words")
    
    with col3:
        total_sources = session_metrics.get('total_sources_retrieved', 0)
        st.metric("📚 Total Sources Used", int(total_sources))
        avg_sources = session_metrics.get('avg_sources_per_response', 0)
        st.metric("🔍 Avg Sources/Response", f"{avg_sources:.1f}")
    
    with col4:
        complexity = session_metrics.get('avg_question_complexity', 0)
        st.metric("🧠 Avg Question Complexity", f"{complexity:.2f}")
        quality = session_metrics.get('avg_response_quality', 0)
        st.metric("⭐ Avg Response Quality", f"{quality:.2f}")
    
    # Current response metrics (if available)
    if current_metrics:
        st.subheader("📈 Latest Response Metrics")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("🎯 Semantic Similarity", f"{current_metrics.get('semantic_similarity', 0):.3f}")
            st.metric("🔗 Response Coherence", f"{current_metrics.get('response_coherence', 0):.3f}")
        
        with col_b:
            st.metric("📋 Question Coverage", f"{current_metrics.get('question_word_coverage', 0):.3f}")
            st.metric("🔍 Source Relevance", f"{current_metrics.get('question_source_relevance', 0):.3f}")
        
        with col_c:
            processing_time = current_metrics.get('processing_time', 0)
            st.metric("⚡ Processing Time", f"{processing_time:.2f}s" if processing_time else "N/A")
            response_speed = current_metrics.get('response_speed', 0)
            st.metric("🏃 Response Speed", f"{response_speed:.1f} words/s" if response_speed else "N/A")
    
    # Detailed analysis in expandable sections
    with st.expander("📊 Detailed Analytics"):
        
        # Create dataframes for visualization
        questions = []
        responses = []
        
        for i, msg in enumerate(chat_history):
            if len(msg) >= 2:
                if msg[0] == 'user':
                    questions.append({
                        'index': len(questions) + 1,
                        'length': len(msg[1].split()),
                        'complexity': evaluator._estimate_question_complexity(msg[1])
                    })
                elif msg[0] == 'assistant':
                    # Find corresponding question
                    q_idx = len(responses)
                    question_text = ""
                    if q_idx < len([m for m in chat_history if m[0] == 'user']):
                        user_msgs = [m[1] for m in chat_history if m[0] == 'user']
                        question_text = user_msgs[q_idx] if q_idx < len(user_msgs) else ""
                    
                    responses.append({
                        'index': len(responses) + 1,
                        'length': len(msg[1].split()),
                        'quality': evaluator._estimate_response_quality(question_text, msg[1]),
                        'sources': len(msg[2]) if len(msg) > 2 and msg[2] else 0
                    })
        
        if questions and responses:
            # Display charts
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.write("**Question Complexity Over Time**")
                q_df = pd.DataFrame(questions)
                st.line_chart(q_df.set_index('index')['complexity'])
                
                st.write("**Question Length Distribution**")
                st.bar_chart(q_df.set_index('index')['length'])
            
            with col_chart2:
                st.write("**Response Quality Over Time**")
                r_df = pd.DataFrame(responses)
                st.line_chart(r_df.set_index('index')['quality'])
                
                st.write("**Sources Used Per Response**")
                st.bar_chart(r_df.set_index('index')['sources'])
    
    # Export metrics
    if st.button("📊 Export Metrics"):
        metrics_data = {
            'session_metrics': session_metrics,
            'current_metrics': current_metrics or {},
            'timestamp': time.time(),
            'questions': questions,
            'responses': responses
        }
        
        json_str = json.dumps(metrics_data, indent=2)
        st.download_button(
            label="📊 Download Metrics as JSON",
            data=json_str,
            file_name=f"rag_metrics_{int(time.time())}.json",
            mime="application/json"
        )