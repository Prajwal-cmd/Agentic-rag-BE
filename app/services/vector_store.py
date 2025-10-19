"""
Qdrant Vector Store Service with Connection Pooling

Pattern: Singleton + Connection Pool (Enterprise Standard)
Source: Microsoft Semantic Kernel, LangChain Production Patterns
Reference: https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors/
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional
from uuid import uuid4
from functools import lru_cache
import threading
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class QdrantConnectionPool:
    """
    Singleton connection pool for Qdrant clients.
    
    Pattern: Connection pooling with lazy initialization
    Reference: https://www.cockroachlabs.com/blog/what-is-connection-pooling/
    """
    
    _instance = None
    _lock = threading.Lock()
    _clients: Dict[str, QdrantClient] = {}
    _collection_cache: Dict[str, bool] = {}  # Cache collection existence
    _cache_lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_client(self, url: str, api_key: str) -> QdrantClient:
        """
        Get or create a pooled Qdrant client.
        
        Pattern: Singleton client per (url, api_key) pair
        """
        client_key = f"{url}:{api_key[:8]}"  # Use partial key for logging
        
        if client_key not in self._clients:
            with self._lock:
                if client_key not in self._clients:
                    logger.info(f"Creating new Qdrant client for {url}")
                    self._clients[client_key] = QdrantClient(
                        url=url, 
                        api_key=api_key,
                        timeout=30  # Add timeout
                    )
        
        return self._clients[client_key]
    
    def collection_exists(self, client_key: str, collection_name: str) -> bool:
        """
        Check if collection exists (with caching).
        
        Pattern: TTL-based cache to avoid repeated API calls
        """
        cache_key = f"{client_key}:{collection_name}"
        
        with self._cache_lock:
            # Return cached value if exists
            if cache_key in self._collection_cache:
                return self._collection_cache[cache_key]
        
        return False
    
    def mark_collection_exists(self, client_key: str, collection_name: str):
        """Mark collection as existing in cache."""
        cache_key = f"{client_key}:{collection_name}"
        with self._cache_lock:
            self._collection_cache[cache_key] = True
            logger.debug(f"Cached collection existence: {collection_name}")
    
    def invalidate_cache(self, client_key: str = None, collection_name: str = None):
        """Invalidate cache (e.g., after collection deletion)."""
        with self._cache_lock:
            if client_key and collection_name:
                cache_key = f"{client_key}:{collection_name}"
                self._collection_cache.pop(cache_key, None)
            else:
                self._collection_cache.clear()
            logger.info("Collection cache invalidated")


# Global connection pool instance
_connection_pool = QdrantConnectionPool()


class VectorStoreService:
    """
    OPTIMIZED: Qdrant vector store with connection pooling and caching.
    
    Changes from original:
    1. Uses shared connection pool instead of creating new clients
    2. Caches collection existence checks
    3. Lazy collection creation (only when needed)
    4. Proper connection lifecycle management
    """
    
    def __init__(
        self, 
        url: str, 
        api_key: str, 
        collection_name: str, 
        embedding_dim: int = 384,
        auto_create: bool = True  # NEW: Control collection creation
    ):
        """
        Initialize vector store with pooled connection.
        
        Args:
            url: Qdrant cluster URL
            api_key: Qdrant API key
            collection_name: Collection name for this session
            embedding_dim: Embedding vector dimension
            auto_create: Automatically create collection if missing (default: True)
        """
        logger.info(f"Initializing vector store for collection: {collection_name}")
        
        # Use shared connection pool instead of creating new client
        self.url = url
        self.api_key = api_key
        self.client_key = f"{url}:{api_key[:8]}"
        self.client = _connection_pool.get_client(url, api_key)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        
        # Only create collection if needed (and auto_create=True)
        if auto_create:
            self._ensure_collection()
    


    def create_payload_indexes(self):
        """
        Create Qdrant payload indexes for fast filtered retrieval.
        
        Indexes contextual metadata fields for instant access.
        Cost: Free, one-time operation per collection.
        """
        try:
            # Index document title
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_title",
                field_schema="keyword"
            )
            
            # Index document type
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_type",
                field_schema="keyword"
            )
            
            # Index chunk position
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chunk_position",
                field_schema="integer"
            )
            
            # Index key entities
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="key_entities",
                field_schema="keyword"
            )
            
            logger.info("✅ Created payload indexes for fast context retrieval")
            
        except Exception as e:
            logger.warning(f"Payload indexing failed (may already exist): {e}")



    def _ensure_collection(self):
        """
        OPTIMIZED: Create collection if it doesn't exist (with caching).
        
        Pattern: Lazy initialization with cache
        Reference: https://refactoring.guru/design-patterns/singleton
        """
        # Check cache first (avoids API call)
        if _connection_pool.collection_exists(self.client_key, self.collection_name):
            logger.debug(f"Collection {self.collection_name} exists (cached)")
            return
        
        try:
            # Optimized: Use get_collection() directly (faster than get_collections())
            try:
                self.client.get_collection(collection_name=self.collection_name)
                # Collection exists - cache it
                _connection_pool.mark_collection_exists(self.client_key, self.collection_name)
                logger.info(f"Collection {self.collection_name} verified (cached)")
                return
            except Exception:
                # Collection doesn't exist - create it
                pass
            
            # Create collection
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            
            # Cache the newly created collection
            _connection_pool.mark_collection_exists(self.client_key, self.collection_name)
            logger.info("Collection created and cached successfully")
            
        except Exception as e:
            # If creation failed but collection exists (race condition), cache it
            try:
                self.client.get_collection(collection_name=self.collection_name)
                _connection_pool.mark_collection_exists(self.client_key, self.collection_name)
                logger.info("Collection exists (race condition handled)")
            except:
                logger.error(f"Error ensuring collection: {e}")
                raise
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None
    ) -> List[str]:
        """Add documents to vector store (unchanged)."""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        points = []
        ids = []
        
        for text, embedding, metadata in zip(texts, embeddings, metadatas):
            point_id = str(uuid4())
            ids.append(point_id)
            
            payload = {
                "text": text,
                **metadata
            }
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )
        
        # Batch upsert
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Added {len(points)} documents to vector store")
        return ids
    
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 5
    ) -> List[Dict]:
        """
        Enhanced similarity search with filename context.
        """
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            with_payload=True
        )
        
        formatted_results = []
        for hit in results:
            result = {
                "text": hit.payload.get("text", ""),
                "metadata": hit.payload,
                "score": hit.score,
                # NEW: Add filename for easy reference
                "filename": hit.payload.get("filename", "Unknown"),
                "source_display": f"{hit.payload.get('filename', 'Unknown')} (chunk {hit.payload.get('chunk_index', '?')})"
            }
            formatted_results.append(result)
        
        return formatted_results
    

    def get_all_documents_for_session(self) -> List[Dict]:
        """
        Get all documents from this session's collection for BM25 indexing.
        
        Returns:
            List of document dicts with text and metadata
        """
        try:
            # Scroll through all points in collection
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Max documents per session
                with_payload=True,
                with_vectors=False  # Don't need vectors for BM25
            )
            
            documents = []
            for point in scroll_result[0]:  # scroll returns (points, next_page_offset)
                doc = {
                    "id": point.id,
                    "text": point.payload.get("text", ""),
                    "metadata": {k: v for k, v in point.payload.items() if k != "text"}
                }
                documents.append(doc)
            
            logger.info(f"Retrieved {len(documents)} documents for BM25 indexing")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents for BM25: {e}")
            return []



    def delete_collection(self):
        """
        OPTIMIZED: Delete collection and invalidate cache.
        """
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            
            # Invalidate cache for this collection
            _connection_pool.invalidate_cache(self.client_key, self.collection_name)
            
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def get_collection_info(self) -> Dict:
        """
        OPTIMIZED: Get collection statistics (with fast-path for existence check).
        """
        try:
            # Fast path: Check cache first
            if not _connection_pool.collection_exists(self.client_key, self.collection_name):
                # Try to get collection info
                try:
                    info = self.client.get_collection(collection_name=self.collection_name)
                except Exception:
                    # Collection doesn't exist
                    return {
                        "name": self.collection_name,
                        "vectors_count": 0,
                        "points_count": 0,
                        "exists": False
                    }
                
                # Collection exists - cache it
                _connection_pool.mark_collection_exists(self.client_key, self.collection_name)
            else:
                # Collection exists in cache - get fresh info
                info = self.client.get_collection(collection_name=self.collection_name)
            
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count if hasattr(info, 'vectors_count') else 0,
                "points_count": info.points_count if hasattr(info, 'points_count') else 0,
                "exists": True
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "name": self.collection_name,
                "vectors_count": 0,
                "points_count": 0,
                "exists": False
            }

    def get_all_session_collections(self) -> List[Dict]:
        """
        Get all session collections with their metadata.
        
        Returns:
            List of dicts with collection names and creation timestamps
        """
        try:
            collections = self.client.get_collections()
            session_collections = []
            
            for collection in collections.collections:
                if collection.name.startswith("session_"):
                    try:
                        # Get collection info to extract creation time
                        info = self.client.get_collection(collection_name=collection.name)
                        
                        session_collections.append({
                            "name": collection.name,
                            "points_count": info.points_count if hasattr(info, 'points_count') else 0,
                            "vectors_count": info.vectors_count if hasattr(info, 'vectors_count') else 0,
                            "session_id": collection.name.replace("session_", "")
                        })
                    except Exception as e:
                        logger.warning(f"Failed to get info for {collection.name}: {e}")
                        continue
            
            logger.info(f"Found {len(session_collections)} session collections")
            return session_collections
            
        except Exception as e:
            logger.error(f"Error listing session collections: {e}")
            return []

    def delete_old_sessions_by_metadata(self, max_age_hours: int = 24) -> Dict:
        """
        Delete session collections older than specified hours using metadata.
        Uses Qdrant's scroll API to check first point timestamp in each collection.
        
        Args:
            max_age_hours: Delete collections older than this many hours
            
        Returns:
            Dict with cleanup statistics
        """
        from datetime import datetime, timedelta
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            cutoff_timestamp = cutoff_time.timestamp()
            
            collections = self.get_all_session_collections()
            
            deleted_count = 0
            skipped_count = 0
            error_count = 0
            
            for collection_info in collections:
                collection_name = collection_info["name"]
                
                try:
                    # Get first point from collection to check timestamp
                    scroll_result = self.client.scroll(
                        collection_name=collection_name,
                        limit=1,
                        with_payload=True,
                        with_vectors=False
                    )
                    
                    points = scroll_result[0]
                    
                    if not points:
                        # Empty collection - delete it
                        logger.info(f"Deleting empty collection: {collection_name}")
                        self.client.delete_collection(collection_name=collection_name)
                        _connection_pool.invalidate_cache(self.client_key, collection_name)
                        deleted_count += 1
                        continue
                    
                    # Check if collection has upload_timestamp in payload
                    first_point = points[0]
                    upload_timestamp = first_point.payload.get("upload_timestamp")
                    
                    if upload_timestamp and upload_timestamp < cutoff_timestamp:
                        logger.info(f"Deleting old collection: {collection_name} (age: {(datetime.now().timestamp() - upload_timestamp) / 3600:.1f}h)")
                        self.client.delete_collection(collection_name=collection_name)
                        _connection_pool.invalidate_cache(self.client_key, collection_name)
                        deleted_count += 1
                    else:
                        skipped_count += 1
                        
                except Exception as e:
                    logger.error(f"Error processing collection {collection_name}: {e}")
                    error_count += 1
                    continue
            
            result = {
                "deleted": deleted_count,
                "skipped": skipped_count,
                "errors": error_count,
                "total_checked": len(collections)
            }
            
            logger.info(f"✅ Session cleanup complete: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Session cleanup failed: {e}")
            return {"deleted": 0, "skipped": 0, "errors": 1, "total_checked": 0}


# Utility function for getting vector store with default settings
@lru_cache(maxsize=128)
def get_cached_vector_store_config(session_id: str, embedding_dim: int):
    """
    Cache vector store configuration.
    
    Pattern: Configuration caching to reduce object creation overhead
    """
    return {
        "session_id": session_id,
        "embedding_dim": embedding_dim,
        "collection_name": f"session_{session_id}"
    }
