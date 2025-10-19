"""
Local Embeddings Service
Pattern: Local inference to eliminate API costs and quota constraints
Source: Sentence-Transformers library (HuggingFace)
Model: all-MiniLM-L6-v2 (384 dims, 80MB, optimized for semantic similarity)
"""
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from ..utils.logger import setup_logger
from functools import lru_cache
from ..config import settings  # ← ADD THIS LINE



logger = setup_logger(__name__)

class EmbeddingService:
    """
    Local embedding generation using sentence-transformers.
    Runs entirely on CPU/GPU without external API calls.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-MiniLM-L3-v2"):
        """
        Initialize lightweight embedding model.
        
        Model: paraphrase-MiniLM-L3-v2
        - Size: 68MB (vs 87MB for L6)
        - Layers: 3 (vs 6 for L6) 
        - Dimensions: 384
        - Speed: 2x faster
        - Accuracy: ~95% of L6 (minimal loss)
        
        Perfect for 512MB deployment - no ONNX needed!
        
        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Loading lightweight embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model loaded. Size: ~68MB, Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            384-dimensional embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        More efficient than individual calls.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()

# Global instance
embedding_service = None
@lru_cache(maxsize=1)

def get_embedding_service(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get embedding service - API or local based on config
    """
    global embedding_service
    
    if embedding_service is None:
        # Check if using API embeddings
        if hasattr(settings, 'use_api_embeddings') and settings.use_api_embeddings:
            from .embeddings_api import get_api_embedding_service
            logger.info("🌐 Using API-based embeddings (zero local memory)")
            embedding_service = get_api_embedding_service()
        else:
            logger.info("💻 Using local embeddings")
            embedding_service = EmbeddingService(model_name)
    
    return embedding_service
