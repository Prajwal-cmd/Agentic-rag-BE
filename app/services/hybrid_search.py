"""
Hybrid Search Service

Pattern: BM25 (lexical) + Semantic (dense) search with Reciprocal Rank Fusion
Source: EMNLP 2024 Best Practices, Microsoft Semantic Kernel

Combines keyword-based and semantic search for better retrieval accuracy.
"""

from typing import List, Dict, Tuple
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import numpy as np
from collections import defaultdict
from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridSearchService:
    """
    Hybrid search combining BM25 (lexical) and semantic (dense) retrieval.
    
    Pipeline:
    1. BM25 search (keyword matching) → top-K results
    2. Semantic search (embedding similarity) → top-K results
    3. Reciprocal Rank Fusion (RRF) to merge results
    4. Return unified ranked list
    """
    
    def __init__(self):
        self.bm25_index = None
        self.corpus = []
        self.documents = []
    
    def build_bm25_index(self, documents: List[Document]):
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of LangChain Document objects
        """
        if not documents:
            logger.warning("No documents provided for BM25 indexing")
            return
        
        try:
            # Extract text and tokenize
            self.documents = documents
            self.corpus = [doc.page_content.lower().split() for doc in documents]
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(self.corpus)
            
            logger.info(f"✅ Built BM25 index with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25_index = None
    
    def bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """
        Perform BM25 search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if self.bm25_index is None or not self.documents:
            logger.warning("BM25 index not built, returning empty results")
            return []
        
        try:
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top-k indices
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # Return documents with scores
            results = [(self.documents[i], scores[i]) for i in top_indices if scores[i] > 0]
            
            logger.debug(f"BM25 search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[Document, float]],
        semantic_results: List[Tuple[Document, float]],
        k: int = 60
    ) -> List[Document]:
        """
        Merge BM25 and semantic results using Reciprocal Rank Fusion.
        
        RRF formula: score(d) = Σ 1/(k + rank(d))
        where k=60 is a constant (standard value from research)
        
        Args:
            bm25_results: BM25 search results
            semantic_results: Semantic search results
            k: RRF constant (default: 60)
            
        Returns:
            Merged and ranked list of documents
        """
        try:
            # Create score dictionary
            rrf_scores = defaultdict(float)
            doc_map = {}
            
            # Add BM25 scores
            for rank, (doc, score) in enumerate(bm25_results):
                doc_id = id(doc)
                rrf_scores[doc_id] += settings.bm25_weight / (k + rank + 1)
                doc_map[doc_id] = doc
            
            # Add semantic scores
            for rank, (doc, score) in enumerate(semantic_results):
                doc_id = id(doc)
                rrf_scores[doc_id] += settings.semantic_weight / (k + rank + 1)
                doc_map[doc_id] = doc
            
            # Sort by RRF score
            sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
            
            # Return merged documents with RRF scores in metadata
            merged_docs = []
            for doc_id in sorted_doc_ids:
                doc = doc_map[doc_id]
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['rrf_score'] = rrf_scores[doc_id]
                merged_docs.append(doc)
            
            logger.info(f"✅ RRF merged {len(merged_docs)} documents")
            return merged_docs
            
        except Exception as e:
            logger.error(f"RRF fusion failed: {e}")
            # Fallback: return semantic results
            return [doc for doc, _ in semantic_results]
    
    def hybrid_search(
        self,
        query: str,
        semantic_results: List[Document],
        top_k: int = 20
    ) -> List[Document]:
        """
        Perform hybrid search combining BM25 and semantic results.
        
        Args:
            query: Search query
            semantic_results: Results from semantic search
            top_k: Number of final results
            
        Returns:
            Merged and ranked documents
        """
        if not settings.enable_hybrid_search:
            logger.debug("Hybrid search disabled, using semantic only")
            return semantic_results[:top_k]
        
        if self.bm25_index is None:
            logger.warning("BM25 index not available, using semantic only")
            return semantic_results[:top_k]
        
        try:
            # Perform BM25 search
            bm25_results = self.bm25_search(query, top_k=top_k)
            
            # Convert semantic results to (doc, score) format
            semantic_with_scores = [
                (doc, doc.metadata.get('score', 0.5))
                for doc in semantic_results[:top_k]
            ]
            
            # Merge with RRF
            merged_docs = self.reciprocal_rank_fusion(
                bm25_results,
                semantic_with_scores
            )
            
            return merged_docs[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return semantic_results[:top_k]


# Global instance
_hybrid_search = None

def get_hybrid_search() -> HybridSearchService:
    """Get or create global hybrid search instance."""
    global _hybrid_search
    if _hybrid_search is None:
        _hybrid_search = HybridSearchService()
    return _hybrid_search
