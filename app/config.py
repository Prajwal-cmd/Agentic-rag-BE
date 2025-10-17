"""
Configuration Management

Source: Pydantic Settings pattern - Industry standard for environment management
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Uses Pydantic for validation and type safety.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )

    # ===== API Keys =====
    groq_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    tavily_api_key: str
    semantic_scholar_api_key: Optional[str] = None  # Optional for Semantic Scholar

    # ===== Model Configuration =====
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    routing_model: str = "llama-3.1-8b-instant"
    grading_model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    generation_model: str = "llama-3.3-70b-versatile"
    
    # NEW: Context generation model for enriching chunks
    context_generation_model: str = "llama-3.1-8b-instant"

    # ===== Vector Store Settings =====
    qdrant_collection_name: str = "rag_documents"

    # ===== Document Processing =====
    chunk_size: int = 500  # CHANGED: Reduced from 1000 for better semantic boundaries
    chunk_overlap: int = 100  # CHANGED: Reduced proportionally
    max_upload_size: int = 15728640  # 15MB in bytes
    
    # NEW: Contextual retrieval settings
    enable_contextual_embedding: bool = True  # Master switch
    
    # NEW: Fast contextual methods
    enable_late_chunking: bool = True  # Jina late chunking (fast)
    enable_template_context: bool = True  # Template-based prefix (instant)
    enable_payload_context: bool = True  # Qdrant metadata indexing (free)

    # ===== Conversation Management =====
    max_messages_before_summary: int = 10
    recent_messages_to_keep: int = 4

    # ===== Retrieval Configuration =====
    retrieval_k: int = 20  # CHANGED: Increased from 4 for reranking pipeline
    retrieval_after_rerank: int = 5  # NEW: Keep top-5 after reranking
    web_search_results: int = 3
    
    # NEW: Hybrid search settings
    enable_hybrid_search: bool = True  # Enable BM25 + semantic search
    bm25_weight: float = 0.3  # Weight for BM25 scoring
    semantic_weight: float = 0.7  # Weight for semantic scoring
    
    # NEW: Reranking settings
    enable_reranking: bool = True  # Enable reranking step
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Lightweight reranker
    
    # NEW: Query rewriting
    enable_query_rewriting: bool = True  # Rewrite vague document queries

    # ===== Research Configuration =====
    research_papers_limit: int = 5  # Max papers to retrieve from Semantic Scholar
    research_citation_threshold: int = 10  # Minimum citations for paper relevance
    research_year_threshold: int = 2018  # Papers after this year preferred

    # ===== NEW: Fallback Configuration =====
    # Web search fallback chain: Tavily → DuckDuckGo → Brave
    enable_web_search_fallback: bool = True
    use_duckduckgo_fallback: bool = True  # Free, unlimited

    # Research fallback chain: Semantic Scholar → arXiv → CORE
    enable_research_fallback: bool = True
    use_arxiv_fallback: bool = True  # Free, unlimited
    use_core_fallback: bool = True  # Free, 1000 req/day
    
    # NEW: Grading Configuration
    relevance_threshold: float = 0.2  # CHANGED: Only fallback if <20% relevant (was 50%)
    min_relevant_docs: int = 1  # CHANGED: Only fallback if zero relevant docs

    # ===== Retry and Error Handling =====
    llm_max_retries: int = 3
    llm_retry_base_delay: float = 1.0
    llm_retry_max_delay: float = 10.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60

    # Fallback model (faster, more reliable for errors)
    fallback_model: str = "llama-3.1-8b-instant"

    # Timeout settings
    llm_timeout: int = 30  # seconds
    api_timeout: int = 60  # seconds

    # NEW: Advanced Router Configuration
    enable_semantic_routing: bool = True  # Enable Tier 2 semantic routing
    semantic_confidence_threshold: float = 0.75  # Minimum confidence for Tier 2
    enable_rag_aware_routing: bool = True  # Enable Tier 3 RAG-aware routing
    routing_log_decisions: bool = True  # Log all routing decisions

settings = Settings()
