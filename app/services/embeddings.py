"""
Jina AI Embeddings Service

Pattern: API-based embedding generation (zero memory footprint)
Source: Jina AI Embeddings API v3
Model: jina-embeddings-v3 (1024 dims, multilingual, 8192 context)
Benefits:
- Zero local memory usage (perfect for 512MB Render limit)
- Better embeddings than all-MiniLM-L6-v2
- 10M free tokens (no credit card needed)
- No automatic billing
"""

import requests
from typing import List, Dict, Any
from ..utils.logger import setup_logger
from ..config import settings

logger = setup_logger(__name__)

class EmbeddingService:
    """
    Jina AI embedding generation via API.
    Zero memory footprint - perfect for constrained deployments.
    """
    
    def __init__(self, model_name: str = "jina-embeddings-v3"):
        """
        Initialize Jina AI embedding service.
        
        Args:
            model_name: Jina model identifier (v3 recommended)
        """
        self.model_name = model_name
        self.api_url = "https://api.jina.ai/v1/embeddings"
        self.api_key = settings.jina_api_key
        
        # Validate API key
        if not self.api_key or not self.api_key.startswith("jina_"):
            raise ValueError("Invalid JINA_API_KEY. Get one from https://jina.ai/embeddings/")
        
        logger.info(f"✅ Jina AI Embeddings initialized: {model_name}")
        logger.info("📊 Model specs: 1024 dims, 8192 context, multilingual")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            1024-dimensional embedding vector
        """
        return self.embed_documents([text])[0]
    
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        Handles up to 2048 texts per request.
        """
        if not texts:
            return []
        
        if len(texts) > 2048:
            logger.warning(f"Batch size {len(texts)} exceeds limit. Splitting into chunks...")
            results = []
            for i in range(0, len(texts), 2048):
                batch = texts[i:i+2048]
                results.extend(self.embed_documents(batch))
            return results
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # ✅ FIXED: Correct parameter names
            payload = {
                "model": self.model_name,
                "input": texts,
                "embedding_type": "float",  # ✅ Changed from "encoding_type"
                "normalized": True  # ✅ Added for better performance
            }
            
            logger.debug(f"Embedding {len(texts)} texts via Jina AI...")
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            # Handle rate limits
            if response.status_code == 429:
                logger.error("❌ Jina API rate limit exceeded (1M TPM)")
                raise Exception("Rate limit exceeded. Wait a moment and try again.")
            
            # ✅ Better 422 error handling
            if response.status_code == 422:
                error_detail = response.json() if response.content else "No details"
                logger.error(f"❌ Jina API 422 Error: {error_detail}")
                raise Exception(f"Invalid request format: {error_detail}")
            
            response.raise_for_status()
            data = response.json()
            
            embeddings = [item["embedding"] for item in data["data"]]
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings
            
        except requests.exceptions.Timeout:
            logger.error("❌ Jina API timeout")
            raise Exception("Embedding API timeout. Try again.")
        
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Jina API error: {e}")
            raise Exception(f"Embedding API error: {str(e)}")


    def get_dimension(self) -> int:
        """Get embedding dimension for jina-embeddings-v3"""
        return 1024  # jina-embeddings-v3 default dimension

# Global instance
_embedding_service = None

def get_embedding_service(model_name: str = "jina-embeddings-v3") -> EmbeddingService:
    """
    Get or create global embedding service instance.
    Singleton pattern to reuse HTTP connections.
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name)
    return _embedding_service
