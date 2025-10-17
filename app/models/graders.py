"""
Structured Output Models for LLM Grading

Pattern: Pydantic models for reliable structured outputs
Source: LangChain structured output pattern
"""

from pydantic import BaseModel, Field
from typing import Literal

class GradeDocuments(BaseModel):
    """
    Binary relevance grading for retrieved documents.
    
    Source: CRAG paper - lightweight retrieval evaluator
    Pattern: LLM-as-a-Judge with binary classification
    """
    
    binary_score: Literal["yes", "no"] = Field(
        description="Document relevance: 'yes' if relevant to question, 'no' otherwise"
    )
    reasoning: str = Field(
        description="Brief explanation of relevance judgment (1-2 sentences)"
    )

class RouteQuery(BaseModel):
    """
    Query routing classification.
    
    Source: Adaptive RAG pattern - semantic query classification
    """
    
    datasource: Literal["vectorstore", "websearch", "hybrid", "research", "hybrid_research"] = Field(
        description="""Route query to appropriate source:
        - vectorstore: Use RAG documents (conceptual, analytical questions)
        - websearch: Use web search (temporal, current events, unknown topics)
        - hybrid: Use both (complex queries needing concepts + current data)
        - research: Use academic paper search (scientific/research questions requiring peer-reviewed sources)
        - hybrid_research: Use documents + research papers (domain-specific questions needing both internal docs and academic context)
        """
    )
    reasoning: str = Field(
        description="Brief explanation of routing decision"
    )
