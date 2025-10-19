"""
LangGraph Workflow Compilation with Dialog Management
Pattern: State machine orchestration with clarification and research support
Source: Official LangGraph documentation + Dialog management + Academic RAG patterns
"""

from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes import (
    route_question,
    retrieve_documents,
    grade_documents,
    transform_query,
    web_search,
    generate,
    direct_llm_generate,
    generate_clarification,
    handle_clarification_response,
    research_search,
    wait_for_upload,
    hybrid_generate,
    hybrid_web_research_generate  # ADD THIS - This is the NEW function
)
from .edges import (
    decide_to_generate,
    route_question_edge,
    clarification_edge
)
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def decide_research_needed(state: dict) -> str:
    """
    After retrieving documents, decide if research papers are needed.
    """
    route_decision = state.get("route_decision", "")
    
    # If route includes "research", fetch academic papers
    if "research" in route_decision:
        logger.info("EDGE: Fetching research papers for hybrid_research route")
        return "research_search"
    else:
        logger.info("EDGE: No research needed, proceeding to grading")
        return "grade_documents"


def compile_workflow() -> StateGraph:
    """
    Compile the CRAG workflow graph with all routing options including hybrid_web_research.
    """
    logger.info("Compiling LangGraph workflow with hybrid_web_research support")
    
    # Initialize graph
    workflow = StateGraph(GraphState)
    
    # Add all nodes
    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve_documents", retrieve_documents)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("direct_llm_generate", direct_llm_generate)
    workflow.add_node("generate_clarification", generate_clarification)
    workflow.add_node("handle_clarification_response", handle_clarification_response)
    workflow.add_node("research_search", research_search)
    workflow.add_node("wait_for_upload", wait_for_upload)
    workflow.add_node("hybrid_generate", hybrid_generate)
    workflow.add_node("hybrid_web_research_generate", hybrid_web_research_generate)
    
    # Set entry point
    workflow.set_entry_point("route_question")
    
    # Add conditional edges from route_question
    workflow.add_conditional_edges(
        "route_question",
        route_question_edge,
        {
            "retrieve_documents": "retrieve_documents",
            "web_search": "web_search",
            "direct_llm_generate": "direct_llm_generate",
            "generate_clarification": "generate_clarification",
            "wait_for_upload": "wait_for_upload",
            "research_search": "research_search",
            "hybrid_web_research_generate": "hybrid_web_research_generate"  # NEW
        }
    )
    
    # After retrieve_documents, check if research is needed
    workflow.add_conditional_edges(
        "retrieve_documents",
        decide_research_needed,
        {
            "research_search": "research_search",
            "grade_documents": "grade_documents"
        }
    )
    
    # After research_search, go to grade_documents
    workflow.add_edge("research_search", "grade_documents")
    
    # Grade documents -> conditional (generate or transform_query)
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate"
        }
    )
    
    # Transform query -> web search
    workflow.add_edge("transform_query", "web_search")
    
    # Web search -> generate
    workflow.add_edge("web_search", "generate")
    
    # Generate -> END
    workflow.add_edge("generate", END)
    
    # Direct LLM generate -> END
    workflow.add_edge("direct_llm_generate", END)
    
    # Wait for upload -> END
    workflow.add_edge("wait_for_upload", END)
    
    # Clarification edges
    workflow.add_edge("generate_clarification", END)
    
    workflow.add_conditional_edges(
        "handle_clarification_response",
        clarification_edge,
        {
            "route_question": "route_question",
            "END": END
        }
    )
    
    # Hybrid generate -> END
    workflow.add_edge("hybrid_generate", END)
    
    # NEW: Hybrid web research generate -> END
    workflow.add_edge("hybrid_web_research_generate", END)
    
    logger.info("Workflow compilation complete with hybrid_web_research support")
    return workflow.compile()


# Factory function
def get_workflow():
    """Get compiled workflow instance."""
    return compile_workflow()
