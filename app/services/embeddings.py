"""
API-based Embeddings Service - Zero server RAM usage
Uses Voyage AI's free tier for embeddings
"""
import requests
from typing import List
import numpy as np
from ..utils.logger import setup_logger
from functools import lru_cache
import os

logger = setup_logger(__name__)

class EmbeddingService:
    """
    API-based embedding generation using Voyage AI.
    No local model loading - zero RAM impact.
    """
    
    def __init__(self, api_key: str = None):
        """Initialize with Voyage AI API key."""
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        self.api_url = "https://api.voyageai.com/v1/embeddings"
        logger.info("✅ Embedding service initialized (API-based, no local model)")
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for single text via API."""
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts via API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "input": texts,
            "model": "voyage-2"  # 1024 dimensions, fast
        }
        
        response = requests.post(self.api_url, json=payload, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        return [item["embedding"] for item in data["data"]]

@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Cached factory for embedding service."""
    return EmbeddingService()
