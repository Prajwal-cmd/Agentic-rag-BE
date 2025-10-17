"""
LangGraph State Definition

Pattern: TypedDict for type-safe state management with dialog management
Source: Official LangGraph documentation + Microsoft Bot Framework patterns
"""

from typing import TypedDict, List, Dict, Optional, Literal
from langchain_core.documents import Document

class GraphState(TypedDict):
    """
    State object passed through the graph workflow.
    Each node can read and update specific fields.
    TypedDict provides type hints for IDE support and validation.
    """
    # User input
    question: str  # Current query (may be transformed)
    original_question: str  # Original query before any transformations
    enriched_question: Optional[str]  # Context-enriched query for persistence
    rewritten_question: Optional[str]  # NEW: Query rewrite for better retrieval
    
    # Retrieved data
    documents: List[Document]  # Documents from RAG or web search
    research_papers: Optional[List[Dict]]  # Research papers from Semantic Scholar
    
    # NEW: Retrieval pipeline state
    initial_documents: Optional[List[Document]]  # Before reranking
    reranked_documents: Optional[List[Document]]  # After reranking
    bm25_documents: Optional[List[Document]]  # BM25 results for hybrid search
    
    # Output
    generation: str  # Final answer
    
    # Control flow flags
    web_search_needed: bool  # Whether to supplement with web search
    research_needed: bool  # Whether to search academic papers
    reranking_applied: bool  # NEW: Track if reranking was applied
    query_rewritten: bool  # NEW: Track if query was rewritten
    
    # Routing
    route_decision: str  # "vectorstore", "web_search", "research", "hybrid", etc.
    
    # Conversation context
    conversation_history: List[Dict]  # Previous messages
    session_id: str  # User session identifier
    
    # Working memory for intermediate results
    working_memory: Dict  # Store intermediate computations
    # NEW: Routing metadata
    routing_confidence: Optional[float]  # Routing confidence score
    routing_tier: Optional[int]  # Which tier made decision (0-3)
    routing_latency_ms: Optional[float]  # Routing decision latency
    routing_metadata: Optional[Dict]  # Additional routing context

    # NEW: Document context awareness
    session_documents: Optional[List[Dict]]  # Metadata of uploaded documents
    document_context: Optional[str]  # Formatted document summary for LLM
    active_document: Optional[str]  # Currently referenced document