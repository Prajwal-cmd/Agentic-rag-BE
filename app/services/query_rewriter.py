"""
Query Rewriting Service

Pattern: Query expansion and rewriting for RAG
Source: EMNLP 2024 Best Practices, RaFe (Ranking Feedback)

Rewrites vague queries to improve retrieval accuracy.
"""

from typing import Dict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from ..config import settings
from ..services.llm_service import get_groq_service
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class RewrittenQuery(BaseModel):
    """Rewritten query with expansion terms."""
    original_query: str = Field(description="Original user query")
    rewritten_query: str = Field(description="Expanded/rewritten query for better retrieval")
    expansion_terms: List[str] = Field(description="Key terms added for expansion")
    reasoning: str = Field(description="Why this rewrite improves retrieval")


class QueryRewriter:
    """
    Rewrites vague or ambiguous queries for better document retrieval.
    
    Examples:
    - "what were the findings" → "research findings results conclusions outcomes discoveries"
    - "summarize it" → "summary overview key points main findings abstract"
    - "compare with recent study" → "comparison analysis differences similarities recent research"
    """
    
    def __init__(self):
        self.groq_service = get_groq_service(settings.groq_api_key)
        self.llm = self.groq_service.get_llm(
            settings.routing_model,
            temperature=0.2
        )
        
        self.rewrite_prompt = ChatPromptTemplate.from_template("""You are a query expansion expert for document retrieval systems.

**Original Query:** {query}

**Task:** Rewrite this query to improve document retrieval by:
1. Expanding abbreviations and vague terms
2. Adding relevant synonyms and related concepts
3. Making implicit aspects explicit
4. Preserving the original intent

**Examples:**

Original: "what were the findings"
Rewritten: "research findings results conclusions outcomes main discoveries key results"

Original: "how does it work"
Rewritten: "methodology approach mechanism implementation process working principle"

Original: "summarize the document"
Rewritten: "summary overview abstract key points main findings executive summary highlights"

**Return JSON format:**
{{
    "original_query": "...",
    "rewritten_query": "...",
    "expansion_terms": ["term1", "term2", ...],
    "reasoning": "..."
}}

Rewrite the query now:""")
    
    def rewrite_query(self, query: str, conversation_context: str = "") -> Dict:
        """
        Rewrite a query for better retrieval.
        
        Args:
            query: Original user query
            conversation_context: Recent conversation context
            
        Returns:
            Dict with rewritten query and metadata
        """
        if not settings.enable_query_rewriting:
            return {
                "original_query": query,
                "rewritten_query": query,
                "expansion_terms": [],
                "reasoning": "Query rewriting disabled"
            }
        
        try:
            # Add conversation context if available
            enriched_query = query
            if conversation_context:
                enriched_query = f"{query} [Context: {conversation_context}]"
            
            # Get rewritten query from LLM
            chain = self.rewrite_prompt | self.llm | StrOutputParser()
            response = chain.invoke({"query": enriched_query})
            
            # Parse JSON response
            import json
            result = json.loads(response)
            
            logger.info(f"Query rewritten: '{query}' → '{result['rewritten_query']}'")
            return result
            
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}. Using original query.")
            return {
                "original_query": query,
                "rewritten_query": query,
                "expansion_terms": [],
                "reasoning": f"Rewriting failed: {str(e)}"
            }
    
    def should_rewrite(self, query: str, route_decision: str) -> bool:
        """
        Determine if query needs rewriting.
        
        Returns True if:
        - Query is vague (very short or contains vague terms)
        - Route decision is vectorstore/hybrid (document queries)
        """
        if route_decision not in ["vectorstore", "hybrid", "hybrid_research"]:
            return False
        
        query_lower = query.lower()
        
        # Vague terms that benefit from expansion
        vague_terms = [
            "findings", "results", "conclusions", "it", "that", "this",
            "methodology", "approach", "summarize", "summary", "compare",
            "difference", "analysis", "explain", "describe", "discuss"
        ]
        
        # Check if query contains vague terms
        has_vague_terms = any(term in query_lower for term in vague_terms)
        
        # Check if query is very short
        is_short = len(query.split()) <= 5
        
        return has_vague_terms or is_short


# Global instance
_query_rewriter = None

def get_query_rewriter() -> QueryRewriter:
    """Get or create global query rewriter instance."""
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = QueryRewriter()
    return _query_rewriter
