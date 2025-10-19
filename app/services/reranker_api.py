"""
API-based Reranking Service using Jina AI
"""
import requests
from typing import List
from langchain_core.documents import Document
from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class APIDocumentReranker:
    def __init__(self):
        self.api_key = settings.jina_api_key
        self.base_url = settings.jina_base_url
        
        if not self.api_key:
            raise ValueError("JINA_API_KEY not found")
        
        logger.info("✅ Using Jina API reranker")
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Document]:
        """Rerank documents via Jina API"""
        
        if not documents:
            return []
        
        if top_k is None:
            top_k = settings.retrieval_after_rerank
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare documents for API
        doc_texts = [doc.page_content[:1000] for doc in documents]
        
        payload = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": doc_texts,
            "top_n": top_k
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/rerank",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            results = data["results"]
            
            # Reorder documents based on API results
            reranked_docs = []
            for item in results:
                idx = item["index"]
                score = item["relevance_score"]
                doc = documents[idx]
                
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['rerank_score'] = score
                doc.metadata['rerank_position'] = len(reranked_docs) + 1
                
                reranked_docs.append(doc)
            
            logger.info(f"✅ Reranked via API: {len(documents)} → {len(reranked_docs)}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Jina rerank API error: {e}")
            return documents[:top_k]

# Global instance
_api_reranker = None

def get_api_reranker() -> APIDocumentReranker:
    global _api_reranker
    if _api_reranker is None:
        _api_reranker = APIDocumentReranker()
    return _api_reranker
