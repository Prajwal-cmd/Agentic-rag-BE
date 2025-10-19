"""
API-based Embeddings Service using Jina AI
Zero local memory footprint - perfect for 512MB deployments
"""
import requests
from typing import List
from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class APIEmbeddingService:
    def __init__(self):
        self.api_key = settings.jina_api_key
        self.base_url = settings.jina_base_url
        self.model = settings.jina_embedding_model
        
        if not self.api_key:
            raise ValueError("JINA_API_KEY not found in environment")
        
        logger.info(f"✅ Using Jina API embeddings: {self.model}")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text via API"""
        return self.embed_documents([text])[0]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via Jina API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "input": texts,
            "encoding_format": "float"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            logger.info(f"✅ Generated {len(embeddings)} embeddings via API")
            return embeddings
            
        except Exception as e:
            logger.error(f"Jina API error: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Jina v3 uses 1024 dimensions, v2 uses 768"""
        return 1024 if "v3" in self.model else 768

# Global instance
_api_embedding_service = None

def get_api_embedding_service() -> APIEmbeddingService:
    global _api_embedding_service
    if _api_embedding_service is None:
        _api_embedding_service = APIEmbeddingService()
    return _api_embedding_service
