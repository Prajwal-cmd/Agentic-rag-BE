"""
Document Reranking Service

Pattern: Cross-encoder reranking for RAG
Source: EMNLP 2024, Anthropic Contextual Retrieval

Reranks retrieved documents using cross-encoder models for improved precision.
Research shows 30-40% improvement in retrieval accuracy.
"""

from typing import List, Dict, Tuple
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from functools import lru_cache
import numpy as np
from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DocumentReranker:
    """
    Reranks retrieved documents using cross-encoder models.
    
    Pipeline:
    1. Retrieve top-K documents (K=20) using bi-encoder (fast)
    2. Rerank with cross-encoder (slow but accurate)
    3. Return top-N (N=5) most relevant
    
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Size: 80MB
    - Speed: ~10ms per query-document pair
    - Accuracy: Better than bi-encoders for relevance
    """
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Lazy load reranking model."""
        if not settings.enable_reranking:
            logger.info("Reranking disabled in config")
            return
        
        try:
            logger.info(f"Loading reranking model: {settings.reranking_model}")
            self.model = CrossEncoder(settings.reranking_model)
            logger.info("✅ Reranking model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranking model: {e}")
            logger.warning("Reranking will be disabled")
            self.model = None
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Document]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: User query
            documents: Retrieved documents to rerank
            top_k: Number of documents to return (default: from config)
            
        Returns:
            Reranked documents sorted by relevance score
        """
        if not settings.enable_reranking or self.model is None:
            logger.debug("Reranking disabled, returning original order")
            return documents[:top_k] if top_k else documents
        
        if not documents:
            return []
        
        if top_k is None:
            top_k = settings.retrieval_after_rerank
        
        try:
            # Prepare query-document pairs
            pairs = []
            for doc in documents:
                # Use page_content for reranking
                doc_text = doc.page_content[:500]  # Limit to 500 chars for speed
                pairs.append([query, doc_text])
            
            # Get cross-encoder scores
            logger.debug(f"Reranking {len(documents)} documents...")
            scores = self.model.predict(pairs)
            
            # Combine documents with scores
            doc_scores = list(zip(documents, scores))
            
            # Sort by score (descending)
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Extract reranked documents
            reranked_docs = [doc for doc, score in doc_scores[:top_k]]
            
            # Add reranking scores to metadata
            for i, (doc, score) in enumerate(doc_scores[:top_k]):
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['rerank_score'] = float(score)
                doc.metadata['rerank_position'] = i + 1
            
            logger.info(f"✅ Reranked {len(documents)} → {len(reranked_docs)} documents")
            logger.debug(f"Top score: {doc_scores[0][1]:.4f}, Bottom score: {doc_scores[-1][1]:.4f}")
            
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            logger.warning("Falling back to original document order")
            return documents[:top_k]
    
    def get_relevance_scores(
        self,
        query: str,
        documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Get relevance scores for all documents without truncating.
        
        Returns:
            List of (document, score) tuples sorted by score
        """
        if not settings.enable_reranking or self.model is None:
            return [(doc, 0.5) for doc in documents]
        
        try:
            pairs = [[query, doc.page_content[:500]] for doc in documents]
            scores = self.model.predict(pairs)
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return doc_scores
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return [(doc, 0.5) for doc in documents]


# Global instance
_reranker = None

def get_reranker() -> DocumentReranker:
    """Get or create global reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = DocumentReranker()
    return _reranker
