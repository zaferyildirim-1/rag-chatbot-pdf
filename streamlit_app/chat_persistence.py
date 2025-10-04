"""
Chat history persistence module for the RAG chatbot.
Allows saving and loading chat sessions with metadata.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import streamlit as st


class ChatHistoryManager:
    """Manages chat history persistence and retrieval."""
    
    def __init__(self, storage_dir: str = "chat_sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
    def save_session(self, 
                    messages: List[Tuple], 
                    session_name: str,
                    document_info: Optional[Dict] = None,
                    chat_stats: Optional[Dict] = None) -> str:
        """
        Save a chat session to disk.
        
        Args:
            messages: List of (role, content) or (role, content, sources) tuples
            session_name: Name for the session
            document_info: Information about the document used
            chat_stats: Statistics about the chat session
            
        Returns:
            Path to saved session file
        """
        
        # Create session data structure
        session_data = {
            "session_name": session_name,
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "document_info": document_info or {},
            "chat_stats": chat_stats or {},
            "version": "1.0"
        }
        
        # Process messages
        for msg in messages:
            if len(msg) >= 2:
                msg_data = {
                    "role": msg[0],
                    "content": msg[1],
                    "timestamp": datetime.now().isoformat()
                }
                
                # Add sources if available
                if len(msg) > 2 and msg[2]:
                    msg_data["sources"] = [
                        {
                            "content": source.page_content[:500],  # Truncate for storage
                            "metadata": source.metadata
                        }
                        for source in msg[2][:3]  # Store only top 3 sources
                    ]
                
                session_data["messages"].append(msg_data)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in session_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{timestamp}_{safe_name}.json"
        filepath = self.storage_dir / filename
        
        # Save to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            return str(filepath)
        except Exception as e:
            st.error(f"Failed to save session: {e}")
            return ""
    
    def load_session(self, filepath: str) -> Optional[Dict]:
        """
        Load a chat session from disk.
        
        Args:
            filepath: Path to session file
            
        Returns:
            Session data dictionary or None if failed
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Failed to load session: {e}")
            return None
    
    def list_sessions(self) -> List[Dict]:
        """
        List all available chat sessions.
        
        Returns:
            List of session metadata dictionaries
        """
        sessions = []
        
        for filepath in self.storage_dir.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract summary info
                session_info = {
                    "filepath": str(filepath),
                    "filename": filepath.name,
                    "session_name": data.get("session_name", "Unnamed Session"),
                    "created_at": data.get("created_at", ""),
                    "message_count": len(data.get("messages", [])),
                    "document_name": data.get("document_info", {}).get("filename", "Unknown"),
                    "file_size": filepath.stat().st_size
                }
                
                sessions.append(session_info)
                
            except Exception as e:
                st.warning(f"Could not read session file {filepath.name}: {e}")
                continue
        
        # Sort by creation date (newest first)
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return sessions
    
    def delete_session(self, filepath: str) -> bool:
        """Delete a session file."""
        try:
            os.remove(filepath)
            return True
        except Exception as e:
            st.error(f"Failed to delete session: {e}")
            return False
    
    def export_session_to_text(self, session_data: Dict) -> str:
        """Export session to readable text format."""
        
        text_lines = []
        text_lines.append(f"Chat Session: {session_data.get('session_name', 'Unnamed')}")
        text_lines.append(f"Created: {session_data.get('created_at', 'Unknown')}")
        text_lines.append(f"Document: {session_data.get('document_info', {}).get('filename', 'Unknown')}")
        text_lines.append("=" * 50)
        text_lines.append("")
        
        for msg in session_data.get("messages", []):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "user":
                text_lines.append(f"USER: {content}")
            else:
                text_lines.append(f"ASSISTANT: {content}")
                
                # Add sources if available
                if "sources" in msg:
                    text_lines.append("  Sources:")
                    for i, source in enumerate(msg["sources"], 1):
                        text_lines.append(f"    {i}. {source['content'][:200]}...")
            
            text_lines.append("")
        
        return "\n".join(text_lines)


def create_session_manager_ui():
    """Create the Streamlit UI for session management."""
    
    # Initialize session manager
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatHistoryManager()
    
    manager = st.session_state.chat_manager
    
    st.subheader("💾 Session Management")
    
    # Save current session
    col1, col2 = st.columns([2, 1])
    
    with col1:
        session_name = st.text_input(
            "Session name", 
            value=f"Chat_{datetime.now().strftime('%Y%m%d_%H%M')}",
            key="session_name_input"
        )
    
    with col2:
        if st.button("💾 Save Session", use_container_width=True):
            if st.session_state.get("msgs") and session_name:
                # Get current document info if available
                doc_info = st.session_state.get("doc_stats", {})
                chat_stats = st.session_state.get("chat_stats", {})
                
                filepath = manager.save_session(
                    st.session_state.msgs,
                    session_name,
                    doc_info,
                    chat_stats
                )
                
                if filepath:
                    st.success(f"✅ Session saved successfully!")
                    st.balloons()
            else:
                st.warning("⚠️ No messages to save or session name is empty")
    
    # Load existing sessions
    st.subheader("📂 Saved Sessions")
    
    sessions = manager.list_sessions()
    
    if not sessions:
        st.info("No saved sessions found.")
        return
    
    # Display sessions in a more compact format
    for i, session in enumerate(sessions):
        with st.expander(
            f"📄 {session['session_name']} "
            f"({session['message_count']} messages, {session['document_name']})",
            expanded=False
        ):
            col_a, col_b, col_c, col_d = st.columns([2, 1, 1, 1])
            
            with col_a:
                st.write(f"**Created:** {session['created_at'][:19]}")
                st.write(f"**Document:** {session['document_name']}")
                st.write(f"**Messages:** {session['message_count']}")
            
            with col_b:
                if st.button("🔄 Load", key=f"load_{i}"):
                    session_data = manager.load_session(session['filepath'])
                    if session_data:
                        # Clear current session
                        st.session_state.msgs = []
                        
                        # Load messages
                        for msg_data in session_data.get("messages", []):
                            role = msg_data.get("role")
                            content = msg_data.get("content")
                            
                            if role and content:
                                if "sources" in msg_data:
                                    # Reconstruct source objects (simplified)
                                    sources = [
                                        type('Source', (), {
                                            'page_content': src['content'],
                                            'metadata': src['metadata']
                                        })()
                                        for src in msg_data["sources"]
                                    ]
                                    st.session_state.msgs.append((role, content, sources))
                                else:
                                    st.session_state.msgs.append((role, content))
                        
                        # Load document info if available
                        if "document_info" in session_data:
                            st.session_state.doc_stats = session_data["document_info"]
                        
                        # Load chat stats if available
                        if "chat_stats" in session_data:
                            st.session_state.chat_stats = session_data["chat_stats"]
                        
                        st.success(f"✅ Loaded session: {session['session_name']}")
                        st.rerun()
            
            with col_c:
                if st.button("📄 Export", key=f"export_{i}"):
                    session_data = manager.load_session(session['filepath'])
                    if session_data:
                        text_export = manager.export_session_to_text(session_data)
                        st.download_button(
                            label="📄 Download as Text",
                            data=text_export,
                            file_name=f"{session['session_name']}.txt",
                            mime="text/plain",
                            key=f"download_{i}"
                        )
            
            with col_d:
                if st.button("🗑️ Delete", key=f"delete_{i}"):
                    if manager.delete_session(session['filepath']):
                        st.success("✅ Session deleted")
                        st.rerun()


def auto_save_session():
    """Automatically save session at regular intervals."""
    
    # This could be called periodically to auto-save sessions
    if "chat_manager" not in st.session_state:
        st.session_state.chat_manager = ChatHistoryManager()
    
    if "msgs" in st.session_state and st.session_state.msgs:
        # Auto-save with timestamp
        auto_name = f"AutoSave_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        manager = st.session_state.chat_manager
        doc_info = st.session_state.get("doc_stats", {})
        chat_stats = st.session_state.get("chat_stats", {})
        
        manager.save_session(
            st.session_state.msgs,
            auto_name,
            doc_info,
            chat_stats
        )