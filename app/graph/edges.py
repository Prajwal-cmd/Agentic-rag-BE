"""
LangGraph Conditional Edges with Clarification Support
Pattern: State-based routing with dialog management
"""

from typing import Dict, Any, Literal
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def decide_to_generate(state: Dict[str, Any]) -> Literal["transform_query", "generate"]:
    """
    FIXED: Decide based on relaxed relevance threshold.
    
    Pattern: CRAG confidence-based branching
    Source: Corrective RAG paper
    
    CRITICAL FIX: Only fallback to web search if relevance is very low (<20%) or zero docs.
    """
    documents = state.get("documents", [])
    web_search_needed = state.get("web_search_needed", False)
    relevance_ratio = state.get("relevance_ratio", 0.0)
    
    # Calculate relevance ratio if not already set
    if documents and relevance_ratio == 0.0:
        total_retrieved = state.get("total_retrieved", len(documents))
        relevant_count = len(documents)
        if total_retrieved > 0:
            relevance_ratio = relevant_count / total_retrieved
        else:
            relevance_ratio = 0.0
    
    logger.info(f"Document relevance: {relevance_ratio:.1%}")
    
    # FIXED: Use new relaxed threshold (20% instead of 50%)
    if web_search_needed or relevance_ratio < 0.2:
        logger.info(f"EDGE: LOW relevance ({relevance_ratio:.0%}) → transform_query + web_search")
        return "transform_query"
    else:
        logger.info(f"EDGE: GOOD relevance ({relevance_ratio:.0%}) → generate")
        return "generate"


def route_question_edge(state: Dict[str, Any]) -> Literal[
    "retrieve_documents",
    "web_search",
    "direct_llm_generate",
    "generate_clarification",
    "wait_for_upload",
    "research_search",
    "hybrid_web_research_generate"  # NEW
]:
    route_decision = state.get("route_decision", "direct_llm_generate")
    logger.info(f"EDGE: route_question_edge -> {route_decision}")
    
    if route_decision == "clarification":
        return "generate_clarification"
    elif route_decision == "wait_for_upload":
        return "wait_for_upload"
    elif route_decision in ["web_search", "websearch"]:
        return "web_search"
    elif route_decision == "vectorstore":
        return "retrieve_documents"
    elif route_decision == "hybrid":
        return "retrieve_documents"
    elif route_decision == "research":
        return "research_search"
    elif route_decision == "hybrid_research":
        return "retrieve_documents"
    elif route_decision == "hybrid_web_research":  # NEW
        return "hybrid_web_research_generate"
    else:
        return "direct_llm_generate"


def clarification_edge(state: Dict[str, Any]) -> Literal["route_question", "END"]:
    """
    Handle clarification response routing.
    
    Args:
        state: Current graph state
    
    Returns:
        Next node name
    """
    dialog_state = state.get("dialog_state", "normal")
    
    if dialog_state == "clarified":
        logger.info("EDGE: Clarification complete, resuming routing")
        return "route_question"
    else:
        logger.info("EDGE: Awaiting user clarification")
        return "END"


def decide_research_hybrid(state: Dict[str, Any]) -> Literal["research_search", "generate"]:
    """
    Decide whether to fetch research papers after retrieving documents.
    Pattern: Multi-source retrieval for comprehensive answers
    
    Args:
        state: Current graph state
    
    Returns:
        Next node name
    """
    route_decision = state.get("route_decision", "")
    research_needed = state.get("research_needed", False)
    
    if "research" in route_decision or research_needed:
        logger.info("EDGE: Fetching research papers for hybrid retrieval")
        return "research_search"
    else:
        logger.info("EDGE: Proceeding to generation")
        return "generate"
