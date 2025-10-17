"""
Document Registry Service

Tracks all uploaded documents per session with metadata.
Pattern: Session-aware document management (LangChain 2024)
"""

from typing import Dict, List, Optional
from datetime import datetime
from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentRegistry:
    """
    In-memory document registry for session-based document tracking.
    
    Stores metadata about uploaded documents to provide context to LLM.
    For production: Replace with Redis/Database for persistence.
    """
    
    def __init__(self):
        # Structure: {session_id: [document_metadata]}
        self._registry: Dict[str, List[Dict]] = {}
        logger.info("✅ Document Registry initialized (in-memory)")
    
    def register_document(
        self,
        session_id: str,
        filename: str,
        chunk_count: int,
        file_size: int,
        file_type: str,
        metadata: Dict = None
    ) -> None:
        """Register a newly uploaded document."""
        if session_id not in self._registry:
            self._registry[session_id] = []
        
        doc_metadata = {
            "filename": filename,
            "chunk_count": chunk_count,
            "file_size": file_size,
            "file_type": file_type,
            "uploaded_at": datetime.now().isoformat(),
            "doc_id": f"{session_id}_{filename}",
            **(metadata or {})
        }
        
        self._registry[session_id].append(doc_metadata)
        logger.info(f"📄 Registered document: {filename} ({chunk_count} chunks)")
    
    def get_session_documents(self, session_id: str) -> List[Dict]:
        """Get all documents for a session."""
        return self._registry.get(session_id, [])
    
    def get_document_summary(self, session_id: str) -> str:
        """Get formatted summary of all session documents."""
        docs = self.get_session_documents(session_id)
        
        if not docs:
            return "No documents uploaded in this session."
        
        summary_parts = [f"📚 {len(docs)} document(s) available:"]
        for i, doc in enumerate(docs, 1):
            summary_parts.append(
                f"{i}. **{doc['filename']}** "
                f"({doc['chunk_count']} chunks, {doc['file_size']/1024:.1f} KB)"
            )
        
        return "\n".join(summary_parts)
    
    def get_latest_document(self, session_id: str) -> Optional[Dict]:
        """Get most recently uploaded document."""
        docs = self.get_session_documents(session_id)
        return docs[-1] if docs else None
    
    def clear_session(self, session_id: str) -> None:
        """Clear all documents for a session."""
        if session_id in self._registry:
            count = len(self._registry[session_id])
            del self._registry[session_id]
            logger.info(f"🗑️ Cleared {count} documents from session {session_id}")


# Global instance
_document_registry = None

def get_document_registry() -> DocumentRegistry:
    """Get or create global document registry."""
    global _document_registry
    if _document_registry is None:
        _document_registry = DocumentRegistry()
    return _document_registry
