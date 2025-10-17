"""
API Request/Response Schemas

Pattern: FastAPI + Pydantic for request validation
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_history: List[Message] = Field(default_factory=list)
    session_id: Optional[str] = None

class Source(BaseModel):
    """Document or research paper source information"""
    content: str  # Excerpt or abstract
    title: str  # Document name or paper title
    url: Optional[str] = None  # Web URL or paper URL
    score: Optional[float] = None  # Relevance score
    type: Literal["vectorstore", "websearch", "research"] = "vectorstore"
    
    # NEW: Research-specific fields
    authors: Optional[List[str]] = None  # Paper authors
    year: Optional[int] = None  # Publication year
    citation_count: Optional[int] = None  # Number of citations
    venue: Optional[str] = None  # Journal/Conference
    paper_id: Optional[str] = None  # Semantic Scholar paper ID

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source] = Field(default_factory=list)
    session_id: str
    route_taken: str  # Which path was used

class UploadResponse(BaseModel):
    message: str
    files_processed: int
    chunks_created: int
    session_id: str

class HealthResponse(BaseModel):
    status: str
    groq_connected: bool
    qdrant_connected: bool
    tavily_connected: bool
    semantic_scholar_connected: bool  # NEW
    embedding_model_loaded: bool
