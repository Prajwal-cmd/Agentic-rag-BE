"""
LangGraph Node Functions - Production Query Routing with Dialog Management

Pattern: Adaptive multi-stage routing with conversational context awareness
Source: SELF-multi-RAG (2024), Agentic RAG Survey (2025)
"""

from typing import Dict, Any, List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from ..models.graders import GradeDocuments, RouteQuery
from ..services.llm_service import get_groq_service
from ..services.embeddings import get_embedding_service
from ..services.vector_store import VectorStoreService
# from ..services.web_search import get_web_search_service
from ..config import settings
from ..utils.logger import setup_logger
# from ..services.research_search import get_research_search_service
from ..services.web_search_fallback import get_web_search_service
from ..services.research_search_fallback import get_research_search_service
# ADD these NEW imports after existing imports
from ..services.query_rewriter import get_query_rewriter
from ..services.reranker import get_reranker
from ..services.hybrid_search import get_hybrid_search
# NEW: Advanced routing system
from ..services.advanced_router import get_advanced_router, RouteType, RoutingDecision

from langchain_core.documents import Document


logger = setup_logger(__name__)

# ========== PYDANTIC MODELS ==========

class EnhancedRouteQuery(BaseModel):
    """Enhanced routing decision with confidence and reasoning."""
    datasource: Literal[
        "vectorstore",
        "web_search",
        "hybrid",
        "direct_llm",
        "research",
        "hybrid_research",
        "hybrid_web_research",
        "continue_with_memory"  # NEW: Use existing context/memory
    ] = Field(
        description="Route decision - use 'continue_with_memory' for follow-ups answerable from conversation"
    )
    
    confidence: float = Field(
        description="Confidence score 0.0-1.0 for this routing decision"
    )
    
    reasoning: str = Field(
        description="Clear explanation for routing choice"
    )
    
    query_complexity: str = Field(
        description="Query complexity: 'simple', 'moderate', 'complex', or 'multi_hop'"
    )
    
    temporal_context: bool = Field(
        description="Whether query requires current/recent information"
    )
    
    needs_clarification: bool = Field(
        default=False,
        description="DEPRECATED: Clarification handled before routing"
    )
    
    clarification_reason: str = Field(
        default="",
        description="DEPRECATED: Clarification handled before routing"
    )

class QueryIntent(BaseModel):
    """Query intent classification for better routing."""
    intent_type: str = Field(
        description="Primary intent: 'factual', 'analytical', 'comparative', 'procedural', 'exploratory', 'computational', 'conversational', 'command', 'follow_up'"
    )
    
    secondary_intents: List[str] = Field(
        default_factory=list,
        description="Secondary intents if query has multiple purposes"
    )
    
    needs_documents: bool = Field(
        description="Whether this query requires document context"
    )
    
    needs_web: bool = Field(
        description="Whether this query requires current web information"
    )
    
    needs_retrieval: bool = Field(
        description="Whether this query needs any external information retrieval"
    )
    
    ambiguity_score: float = Field(
        description="Query ambiguity score 0.0-1.0 (0=clear, 1=very ambiguous)"
    )
    
    context_dependent: bool = Field(
        description="Whether query depends on previous context"
    )
    
    is_follow_up: bool = Field(
        default=False,
        description="Whether this is a follow-up query referencing previous conversation"
    )

class QueryClarification(BaseModel):
    """Clarification requirements for ambiguous queries."""
    needs_clarification: bool = Field(
        description="Whether clarification is needed"
    )
    
    clarification_type: str = Field(
        description="Type: 'missing_documents', 'ambiguous_intent', 'missing_context', 'multiple_interpretations', 'permission_needed'"
    )
    
    clarification_message: str = Field(
        description="Message to present to user for clarification"
    )
    
    suggested_options: List[str] = Field(
        default_factory=list,
        description="Suggested options for user to choose from"
    )
    
    confidence_without_clarification: float = Field(
        description="Confidence score if proceeding without clarification"
    )



# ========== CONTEXT ENRICHMENT FUNCTIONS ==========



def summarize_conversation_for_query(
    question: str,
    history: List[Dict],
    working_memory: Dict[str, str],
    lookback: int = 3
) -> str:
    """
    Summarize conversation context into enriched query.
    Pattern: SELF-multi-RAG query summarization (proven 13.5% retrieval improvement)
    
    Returns enriched query that includes relevant context.
    """
    if not history:
        return question
    
    # Get recent conversation turns
    recent_turns = history[-lookback:]
    
    # Build context summary
    context_parts = []
    
    # Add working memory variables
    if working_memory:
        context_parts.append("Known values: " + ", ".join([f"{k}={v}" for k, v in working_memory.items()]))
    
    # Extract key topics from recent conversation
    topics = []
    for turn in recent_turns:
        content = turn.get("content", "")
        # Extract entities, numbers, and key phrases
        entities = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', content)
        numbers = re.findall(r'\d+(?:\.\d+)?', content)
        topics.extend(entities[:2])  # Limit to 2 entities per turn
        topics.extend(numbers[:2])
    
    # Add recent topics to context
    if topics:
        context_parts.append("Recent context: " + ", ".join(topics[-5:]))
    
    # Build enriched query
    if context_parts:
        enriched = f"{question} (Context: {' | '.join(context_parts)})"
        logger.info(f"Enriched query: {enriched}")
        return enriched
    
    return question








def extract_recent_topic(history: List[Dict], lookback: int = 3) -> str:
    """Extract the main topic/entity from recent conversation."""
    if not history:
        return ""
    
    recent_messages = history[-lookback:]
    topics = []
    
    for msg in recent_messages:
        content = msg.get("content", "")
        # Extract equations, variables, and main entities
        entities = re.findall(r'[a-z]\s*=\s*[^,\s]+|\d+|[A-Z][a-z]+', content)
        topics.extend(entities)
    
    return ", ".join(topics[-3:]) if topics else ""




def format_working_memory_context(working_memory: Dict[str, str]) -> str:
    """
    Format working memory for injection into LLM prompts.
    Pattern: LangChain ConversationBufferMemory formatting
    """
    if not working_memory:
        return ""
    
    context_parts = []
    for key, value in working_memory.items():
        context_parts.append(f"{key} = {value}")
    
    return "\n".join([
        "**Working Memory (Variables and Context):**",
        "\n".join(context_parts),
        ""
    ])


# ========== QUERY COMPLETENESS DETECTION ==========

def detect_query_completeness(
    question: str,
    intent: QueryIntent,
    history: List[Dict],
    has_documents: bool,
    follow_up_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    FIXED: Detect if query is complete, incomplete, or requires disambiguation.
    Pattern: Incomplete Utterance Detection from Microsoft Research & Rasa Pro
    
    CRITICAL FIX: Follow-up queries should NOT require clarification if context exists
    
    Returns:
        completeness_type: 'complete', 'incomplete', 'command', 'ambiguous', 'missing_documents', 'follow_up_complete'
        requires_clarification: bool
        clarification_reason: str
    """
    question_lower = question.lower().strip()
    
    # **PRIORITY 0: Check if this is a follow-up that can be answered from context**
    if follow_up_info.get("is_follow_up") and follow_up_info.get("can_answer_from_memory"):
        return {
            "completeness_type": "follow_up_complete",
            "requires_clarification": False,
            "clarification_reason": "",
            "confidence": 0.95
        }
    
    # **PRIORITY 1: Check if query explicitly needs documents but none available**
    doc_keywords = [
        'document', 'paper', 'file', 'pdf', 'uploaded', 'attachment',
        'doc', 'summarize', 'summary', 'analyze', 'review',
        'the document', 'the paper', 'the file', 'my document',
        'this document', 'these documents'
    ]
    
    mentions_documents = any(keyword in question_lower for keyword in doc_keywords)
    
    if (mentions_documents or intent.needs_documents) and not has_documents:
        general_doc_questions = [
            'what is a document',
            'what are documents',
            'how to upload',
            'how do i upload',
            'can i upload',
            'document format'
        ]
        
        is_general_question = any(gq in question_lower for gq in general_doc_questions)
        
        if not is_general_question:
            return {
                "completeness_type": "missing_documents",
                "requires_clarification": True,
                "clarification_reason": "Query requires documents but none are uploaded",
                "confidence": 0.9
            }
    
    # CASE 1: Explicit commands (complete and actionable)
    command_patterns = [
        r'^search (the |for )?web',
        r'^(google|bing|search) (for )?',
        r'^look up',
        r'^find (me )?information',
        r'^what is the weather',
        r'^calculate',
        r'^tell me about \w+',
    ]
    
    if any(re.match(pattern, question_lower) for pattern in command_patterns):
        if len(question.split()) <= 3:
            return {
                "completeness_type": "incomplete_command",
                "requires_clarification": True,
                "clarification_reason": "Command missing subject/topic",
                "confidence": 0.8
            }
        
        return {
            "completeness_type": "command",
            "requires_clarification": False,
            "clarification_reason": "",
            "confidence": 0.9
        }
    
    # CASE 2: Incomplete queries (missing critical information)
    incomplete_indicators = [
        r'^(what|how|why|when|where|who)\??$',
        r'^(tell me|show me|explain|describe)\??$',
        r'^(the|a|an|this|that|it)\s',
        r'\b(about|of|for|in)\s*$',
        r'^[a-z]{1,2}[\+\-\*/]$',
    ]
    
    if any(re.match(pattern, question_lower) for pattern in incomplete_indicators):
        return {
            "completeness_type": "incomplete",
            "requires_clarification": True,
            "clarification_reason": "Query is incomplete or missing subject",
            "confidence": 0.85
        }
    
    # CASE 3: Ambiguous references ONLY if no history
    if not history or len(history) < 2:
        ambiguous_patterns = [
            r'^(it|this|that|the one|these|those)',
            r'\b(it|this|that)\b.*\?$',
            r'^(what about|how about|why|explain)\s+(it|this|that|the)',
        ]
        
        if any(re.match(pattern, question_lower) for pattern in ambiguous_patterns):
            return {
                "completeness_type": "ambiguous_reference",
                "requires_clarification": True,
                "clarification_reason": "Ambiguous reference without prior context",
                "confidence": 0.8
            }
    
    # CASE 4: High ambiguity from intent classification (FIXED: threshold lowered to 0.8)
    if intent.ambiguity_score >= 0.8:
        return {
            "completeness_type": "ambiguous",
            "requires_clarification": True,
            "clarification_reason": "High semantic ambiguity",
            "confidence": 1.0 - intent.ambiguity_score
        }
    
    # CASE 5: Complete query
    return {
        "completeness_type": "complete",
        "requires_clarification": False,
        "clarification_reason": "",
        "confidence": 0.9
    }


def generate_smart_clarification(
    original_question: str,
    enriched_question: str,
    completeness: Dict[str, Any],
    intent: QueryIntent,
    has_documents: bool,
    working_memory: Dict[str, str]
) -> Dict[str, Any]:
    """
    Generate context-aware clarification messages.
    Pattern: Smart clarification from IBM Watson & LivePerson
    """
    completeness_type = completeness["completeness_type"]
    
    # Type 0: Missing documents (ADDED)
    if completeness_type == "missing_documents":
        return {
            "message": f"You asked to '{original_question}', but no documents are currently uploaded. Would you like to:",
            "options": [
                "Upload a document first",
                "Search the web instead"
            ]
        }
    
    # Type 1: Incomplete command
    if completeness_type == "incomplete_command":
        return {
            "message": f"I see you want to search the web. What would you like me to search for?",
            "options": [
                "Search for general information",
                "Search for recent news",
                "Search for technical information"
            ]
        }
    
    # Type 2: Incomplete query
    if completeness_type == "incomplete":
        return {
            "message": f"Your question seems incomplete: '{original_question}'. Could you provide more details?",
            "options": [
                "Tell me what you want to know",
                "Provide the topic or subject"
            ]
        }
    
    # Type 3: Ambiguous reference
    if completeness_type == "ambiguous_reference":
        return {
            "message": f"You mentioned '{original_question}', but I'm not sure what you're referring to. Could you clarify?",
            "options": [
                "Specify what 'it' or 'this' refers to",
                "Start a new question"
            ]
        }
    
    # Type 4: General high ambiguity
    return {
        "message": f"I'm not sure I understand: '{original_question}'. Could you rephrase or provide more context?",
        "options": []
    }


def should_clarify_or_resolve(
    question: str,
    enriched_question: str,
    has_documents: bool,
    intent: QueryIntent,
    context_switch: Dict[str, Any],
    working_memory: Dict[str, str]
) -> Dict[str, Any]:
    """
    FIXED: Decide: (1) Clarify, (2) Auto-resolve with context, or (3) Proceed
    Pattern: Three-tier decision framework from enterprise dialog systems
    
    CRITICAL FIX: Low ambiguity (<0.7) should NEVER require clarification
    """
    ambiguity_score = intent.ambiguity_score
    
    # TIER 1: High confidence - proceed directly (FIXED: threshold at 0.7)
    if ambiguity_score < 0.7 and not (intent.needs_documents and not has_documents):
        return {
            "action": "proceed",
            "resolved_query": enriched_question,
            "confidence": 1.0 - ambiguity_score
        }
    
    # TIER 2: Medium ambiguity - attempt auto-resolution
    if 0.3 <= ambiguity_score < 0.7:
        # Check if working memory or history can resolve
        if enriched_question != question:  # Context was added
            return {
                "action": "proceed_with_context",
                "resolved_query": enriched_question,
                "confidence": 0.7,
                "note": "Resolved using conversation context"
            }
        
        # Check if it's a computational follow-up
        if intent.intent_type == "computational" and working_memory:
            return {
                "action": "proceed_with_memory",
                "resolved_query": question,
                "confidence": 0.75
            }
        
        # Handle missing documents
        if intent.needs_documents and not has_documents:
            # Check if query explicitly mentions documents
            doc_keywords = ['document', 'paper', 'file', 'pdf', 'the doc', 'uploaded', 'attachment']
            mentions_docs = any(keyword in question.lower() for keyword in doc_keywords)
            
            if mentions_docs:
                # User explicitly wants documents
                return {
                    "action": "clarify",
                    "reason": "missing_documents",
                    "confidence": 0.3
                }
            else:
                # General query - use web search
                return {
                    "action": "soft_clarification",
                    "message": f"I'll search the web for: '{question}'",
                    "fallback_route": "web_search"
                }
    
    # TIER 3: High ambiguity - clarification needed
    if ambiguity_score >= 0.7:
        return {
            "action": "clarify",
            "reason": "high_ambiguity",
            "confidence": 1.0 - ambiguity_score
        }
    
    # Default: proceed with best effort
    return {
        "action": "proceed",
        "resolved_query": enriched_question,
        "confidence": 0.5
    }


def add_capability_context_to_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add capability awareness to state for LLM transparency.
    Pattern: Capability-aware prompting for better user experience
    """
    try:
        embedding_service = get_embedding_service()
        vector_store = VectorStoreService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=f"session_{state['session_id']}",
            embedding_dim=embedding_service.get_dimension()
        )
        
        collection_info = vector_store.get_collection_info()
        has_documents = collection_info.get("points_count", 0) > 0
    except Exception as e:
        logger.warning(f"Could not check document availability: {e}")
        has_documents = False
    
    capabilities_note = f"""**SYSTEM CAPABILITIES AVAILABLE TO YOU:**
- Web search: ✓ ACTIVE (you can search the internet)
- Document search: {"✓ ACTIVE (documents uploaded)" if has_documents else "✗ INACTIVE (no documents uploaded yet)"}
- Conversation memory: ✓ ACTIVE (you remember previous context)
- Computational reasoning: ✓ ACTIVE (you can solve math problems)

**CRITICAL: Never tell users you cannot search the web or access information. You have these capabilities through the system.**"""
    
    state["system_capabilities"] = capabilities_note
    return state


# ========== EXISTING HELPER FUNCTIONS ==========

def detect_context_switch(current_question: str, history: List[Dict]) -> Dict[str, Any]:
    """
    Detect if user has switched context from previous conversation.
    Pattern: Context coherence checking from DialogFlow
    """
    if not history:
        return {"switch_detected": False, "confidence": 1.0}
    
    # Get last few exchanges for context
    recent_context = " ".join([
        msg.get("content", "") for msg in history[-4:]
        if msg.get("role") == "user"
    ])
    
    if not recent_context:
        return {"switch_detected": False, "confidence": 1.0}
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze if the current question represents a context switch from the previous conversation.

Context switch indicators:
- Completely new topic unrelated to previous discussion
- Sudden change in domain (e.g., from technical to personal)
- Ignoring previous clarification request
- Starting fresh without acknowledging previous context

Score:
- 0.0-0.3: Same context, natural continuation
- 0.3-0.6: Related but shifting focus
- 0.6-0.8: Probable context switch
- 0.8-1.0: Definite context switch

Return a score and brief explanation."""),
        ("human", """Previous context: {recent_context}

Current question: {current_question}

Analyze context switch:""")
    ])
    
    groq_service = get_groq_service(settings.groq_api_key)
    llm = groq_service.get_llm(settings.routing_model, temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({
            "recent_context": recent_context,
            "current_question": current_question
        })
        
        # Parse score from result
        score_match = re.search(r"(\d+\.?\d*)", result)
        score = float(score_match.group(1)) if score_match else 0.5
        
        return {
            "switch_detected": score > 0.6,
            "confidence": score,
            "explanation": result
        }
    
    except Exception as e:
        logger.error(f"Context switch detection error: {e}")
        return {"switch_detected": False, "confidence": 0.5}


def analyze_query_for_clarification(
    question: str,
    has_documents: bool,
    intent: QueryIntent,
    context_switch: Dict[str, Any]
) -> QueryClarification:
    """
    Analyze if query needs clarification before processing.
    Pattern: Clarification dialog management from Microsoft Bot Framework
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at identifying when user clarification is needed.

**Context:**
- Documents available: {has_documents}
- Query ambiguity: {ambiguity_score}
- Context switch detected: {context_switch}
- Query needs documents: {needs_documents}

**Clarification Triggers:**
1. **missing_documents**: User asks about "the document/paper/file" but none uploaded
2. **ambiguous_intent**: Query could mean multiple things
3. **missing_context**: References something unclear ("it", "that", "the thing")
4. **multiple_interpretations**: Query has 2+ valid interpretations
5. **permission_needed**: Action requires explicit user confirmation

**Rules:**
- Only suggest clarification for genuinely ambiguous cases
- If query is clear despite missing resources, proceed with appropriate fallback
- Provide actionable options (not just "please clarify")
- Be concise and helpful in clarification messages

**Examples:**
"What's in the document?" + no docs → needs clarification with upload/search options
"What is machine learning?" + no docs → NO clarification (general knowledge)
"Analyze the methodology" + no docs → needs clarification (which methodology?)
"Tell me about it" + no context → needs clarification (about what?)"""),
        ("human", """Query: {question}

Determine if clarification needed:""")
    ])
    
    groq_service = get_groq_service(settings.groq_api_key)
    structured_llm = groq_service.get_structured_llm(
        settings.routing_model,
        QueryClarification
    )
    
    chain = prompt | structured_llm
    return chain.invoke({
        "question": question,
        "has_documents": has_documents,
        "ambiguity_score": intent.ambiguity_score,
        "context_switch": context_switch.get("switch_detected", False),
        "needs_documents": intent.needs_documents
    })


def check_document_availability(session_id: str) -> bool:
    """Check if documents are available for this session."""
    try:
        embedding_service = get_embedding_service()
        vector_store = VectorStoreService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=f"session_{session_id}",
            embedding_dim=embedding_service.get_dimension()
        )
        
        collection_info = vector_store.get_collection_info()
        points_count = collection_info.get("points_count", 0)
        logger.info(f"Document check: session_{session_id} has {points_count} documents")
        return points_count > 0
    except Exception as e:
        logger.warning(f"Could not check document availability: {e}")
        return False

def intelligent_llm_routing(
    question: str,
    enriched_question: str,
    has_documents: bool,
    intent: QueryIntent,
    follow_up_info: Dict[str, Any]
) -> EnhancedRouteQuery:
    """
    FIXED: Intelligent LLM routing with follow-up awareness.
    Only used when rule-based and semantic routing fail.
    """
    try:
        logger.info("Using LLM routing for ambiguous case")
        
        # Build context-aware prompt
        prompt = ChatPromptTemplate.from_template("""
You are an intelligent query router for a RAG system.

Route this query to ONE of these sources:
- vectorstore: User has documents uploaded and query asks about them
- web_search: Needs current/online information (news, weather, recent events)
- direct_llm: General knowledge, greetings, or can answer from conversation memory
- research: Academic papers needed
- hybrid: Documents + web needed
- hybrid_web_research: Research papers + web needed
- continue_with_memory: Follow-up query that can be answered using previous conversation/calculations

Query: {question}
Enriched Query: {enriched_question}
Has documents: {has_documents}
Intent type: {intent_type}
Is follow-up: {is_follow_up}
Can answer from memory: {can_answer_from_memory}

CRITICAL: If is_follow_up=True and can_answer_from_memory=True, choose "continue_with_memory" or "direct_llm", NOT web_search.

Choose the BEST single route and explain briefly why.""")
        
        groq_service = get_groq_service(settings.groq_api_key)
        
        try:
            structured_llm = groq_service.get_structured_llm(
                settings.routing_model,
                EnhancedRouteQuery
            )
            
            chain = prompt | structured_llm
            
            result = groq_service.invoke_with_fallback(
                chain,
                {
                    "question": question,
                    "enriched_question": enriched_question,
                    "has_documents": has_documents,
                    "intent_type": intent.intent_type,
                    "is_follow_up": follow_up_info.get("is_follow_up", False),
                    "can_answer_from_memory": follow_up_info.get("can_answer_from_memory", False)
                },
                schema=EnhancedRouteQuery
            )
            
            # Validate route
            valid_routes = ["vectorstore", "web_search", "hybrid", "direct_llm", "research", "hybrid_research", "hybrid_web_research", "continue_with_memory"]
            if result.datasource not in valid_routes:
                raise ValueError(f"Invalid route: {result.datasource}")
            
            return result
        
        except Exception as llm_error:
            logger.error(f"LLM routing failed: {llm_error}, using intelligent fallback")
            
            # INTELLIGENT FALLBACK LOGIC
            if follow_up_info.get("can_answer_from_memory"):
                return EnhancedRouteQuery(
                    datasource="direct_llm",
                    confidence=0.7,
                    reasoning="Fallback: follow-up query with memory context",
                    query_complexity="simple",
                    temporal_context=False
                )
            elif has_documents:
                return EnhancedRouteQuery(
                    datasource="hybrid",
                    confidence=0.6,
                    reasoning="Fallback: has documents, using hybrid search",
                    query_complexity="moderate",
                    temporal_context=False
                )
            else:
                return EnhancedRouteQuery(
                    datasource="web_search",
                    confidence=0.6,
                    reasoning="Fallback: no documents, using web search",
                    query_complexity="moderate",
                    temporal_context=True
                )
    
    except Exception as e:
        logger.error(f"All routing attempts failed: {e}, using safe fallback")
        
        # ULTIMATE SAFE FALLBACK
        return EnhancedRouteQuery(
            datasource="web_search" if not has_documents else "hybrid",
            confidence=0.5,
            reasoning="Emergency fallback due to routing failures",
            query_complexity="unknown",
            temporal_context=False
        )


def apply_fallback_logic(
    routing_decision: EnhancedRouteQuery,
    has_documents: bool,
    intent: QueryIntent
) -> str:
    """
    Apply fallback logic to ensure valid routing.
    Pattern: Defensive routing with graceful degradation
    """
    route = routing_decision.datasource
    
    # CRITICAL FIX: Override any "clarification" route
    if route == "clarification":
        logger.error("CRITICAL BUG: 'clarification' returned as route - overriding to 'web_search'")
        return "web_search"
    
    # Fallback 1: Can't route to vectorstore if no documents
    if route == "vectorstore" and not has_documents:
        logger.warning("Routing changed: vectorstore → web_search (no documents)")
        return "web_search"
    
    # Fallback 2: Can't route to hybrid if no documents
    if route == "hybrid" and not has_documents:
        logger.warning("Routing changed: hybrid → web_search (no documents)")
        return "web_search"
    
    # Fallback 3: Can't route to hybrid_research if no documents
    if route == "hybrid_research" and not has_documents:
        logger.warning("Routing changed: hybrid_research → research (no documents)")
        return "research"
    
    # NEW Fallback 4: hybrid_web_research always works (no document dependency)
    # No fallback needed - both research and web search are always available
    
    # Fallback 5: If computational/conversational, prefer direct_llm
    if intent.intent_type in ["computational", "conversational"] and route in ["web_search", "vectorstore"]:
        if not intent.needs_retrieval:
            logger.info("Routing optimized: Using direct_llm for computational/conversational")
            return "direct_llm"
    
    return route


def extract_working_memory(question: str, history: List[Dict]) -> Dict[str, str]:
    """
    Extract variables, facts, and context from conversation.
    Pattern: Working memory extraction from dialog systems
    """
    # Look at current message + last 5 messages
    recent_messages = [question] + [
        msg.get("content", "") for msg in history[-5:]
    ]
    
    working_memory = {}
    
    # Extract variable assignments (x = 5, name: John, etc.)
    for message in recent_messages:
        # Pattern 1: x = value
        assignments = re.findall(r'([a-zA-Z_]\w*)\s*=\s*([^,\n]+)', message)
        for var, value in assignments:
            working_memory[var.strip()] = value.strip()
        
        # Pattern 2: variable: value
        colon_assignments = re.findall(r'([a-zA-Z_]\w*)\s*:\s*([^,\n]+)', message)
        for var, value in colon_assignments:
            if var.lower() not in ['http', 'https']:  # Avoid URLs
                working_memory[var.strip()] = value.strip()
    
    if working_memory:
        logger.info(f"Extracted working memory: {working_memory}")
    
    return working_memory


# ========== MAIN NODE FUNCTIONS (WITH CRITICAL FIXES) ==========

"""
CRITICAL FIXES for nodes.py - Add to existing file

Pattern: Defense in Depth with Rule-Based Pre-Filtering
Source: Production RAG Systems (Anthropic, OpenAI)
"""

def classify_query_intent(question: str) -> QueryIntent:
    """
    FIXED: Classify query intent with error handling and fallback.
    
    Pattern: Rule-based pre-filtering before LLM classification
    Source: Agentic RAG Best Practices (2025)
    """
    
    # STEP 1: Rule-based quick classification (no LLM needed)
    question_lower = question.lower().strip()
    
    # Simple patterns that don't need LLM
    if len(question) < 3:
        return QueryIntent(
            intent_type="conversational",
            needs_documents=False,
            needs_web=False,
            needs_retrieval=False,
            ambiguity_score=1.0,
            context_dependent=True,
            secondary_intents=[]
        )
    
    # Math/computation patterns
    if any(op in question for op in ['+', '-', '*', '/', '=', 'calculate', 'compute']):
        return QueryIntent(
            intent_type="computational",
            needs_documents=False,
            needs_web=False,
            needs_retrieval=False,
            ambiguity_score=0.2,
            context_dependent=False,
            secondary_intents=[]
        )
    
    # Document-specific patterns
    doc_keywords = ['document', 'uploaded', 'file', 'pdf', 'the doc', 'this file']
    if any(kw in question_lower for kw in doc_keywords):
        return QueryIntent(
            intent_type="factual",
            needs_documents=True,
            needs_web=False,
            needs_retrieval=True,
            ambiguity_score=0.1,
            context_dependent=False,
            secondary_intents=[]
        )
    
    # Web search commands
    web_keywords = ['search the web', 'google', 'look up online', 'search for']
    if any(kw in question_lower for kw in web_keywords):
        return QueryIntent(
            intent_type="command",
            needs_documents=False,
            needs_web=True,
            needs_retrieval=True,
            ambiguity_score=0.1,
            context_dependent=False,
            secondary_intents=[]
        )
    
    # STEP 2: LLM classification with error handling (only if needed)
    try:
        # SIMPLIFIED PROMPT - much more robust
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Classify this query into ONE intent type:
- factual: Facts/definitions (e.g., "What is X?")
- analytical: Analysis needed
- comparative: Comparing things
- procedural: How-to questions
- exploratory: Open research
- computational: Math/calculations
- conversational: Greetings/chat
- command: Action requests

Also determine:
- needs_documents: true if query mentions uploaded documents
- needs_web: true if query needs current/online information
- ambiguity_score: 0.0 (clear) to 1.0 (very unclear)

Return JSON with: intent_type, needs_documents, needs_web, needs_retrieval, ambiguity_score, context_dependent"""),
            ("human", "Query: {question}")
        ])
        
        groq_service = get_groq_service(settings.groq_api_key)
        
        # Try structured output first
        try:
            structured_llm = groq_service.get_structured_llm(
                settings.routing_model,
                QueryIntent
            )
            chain = prompt | structured_llm
            result = groq_service.invoke_with_fallback(
                chain,
                {"question": question},
                schema=QueryIntent
            )
            return result
            
        except Exception as structured_error:
            logger.warning(f"Structured classification failed: {structured_error}, using fallback")
            
            # FALLBACK: Use regular LLM + manual parsing
            llm = groq_service.get_llm(settings.routing_model, temperature=0)
            chain = prompt | llm | StrOutputParser()
            
            response = chain.invoke({"question": question})
            
            # Parse manually
            import json
            try:
                # Try to extract JSON from response
                json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    return QueryIntent(**data)
            except:
                pass
            
            # Ultimate fallback: heuristic classification
            return QueryIntent(
                intent_type="exploratory",
                needs_documents=False,
                needs_web=True,
                needs_retrieval=True,
                ambiguity_score=0.5,
                context_dependent=False,
                secondary_intents=[]
            )
    
    except Exception as e:
        logger.error(f"All intent classification attempts failed: {e}")
        
        # SAFE FALLBACK: Default to web search
        return QueryIntent(
            intent_type="exploratory",
            needs_documents=False,
            needs_web=True,
            needs_retrieval=True,
            ambiguity_score=0.5,
            context_dependent=False,
            secondary_intents=[]
        )


def make_routing_decision(question: str, has_documents: bool, intent: QueryIntent) -> EnhancedRouteQuery:
    """
    FIXED: Make routing decision with robust error handling.
    
    Pattern: Rule-based routing with LLM as tiebreaker only
    Source: Production Agentic RAG (Microsoft, Google)
    """
    
    # RULE-BASED ROUTING (80% of queries can be handled without LLM)
    
    # Rule 1: Computational queries → direct_llm
    if intent.intent_type == "computational":
        return EnhancedRouteQuery(
            datasource="direct_llm",
            confidence=0.95,
            reasoning="Computational query handled by LLM directly",
            query_complexity="simple",
            temporal_context=False
        )
    
    # Rule 2: Explicit document queries with documents → vectorstore
    if intent.needs_documents and has_documents:
        return EnhancedRouteQuery(
            datasource="vectorstore",
            confidence=0.9,
            reasoning="Query explicitly needs documents and documents are available",
            query_complexity="moderate",
            temporal_context=False
        )
    
    # Rule 3: Explicit document queries WITHOUT documents → web_search
    if intent.needs_documents and not has_documents:
        return EnhancedRouteQuery(
            datasource="web_search",
            confidence=0.85,
            reasoning="Query needs documents but none available, using web search",
            query_complexity="moderate",
            temporal_context=False
        )
    
    # Rule 4: Explicit web search commands → web_search
    if intent.intent_type == "command" and intent.needs_web:
        return EnhancedRouteQuery(
            datasource="web_search",
            confidence=0.95,
            reasoning="Explicit web search command",
            query_complexity="simple",
            temporal_context=True
        )
    
    # Rule 5: Research queries → hybrid_web_research
    research_keywords = ['research', 'study', 'studies', 'paper', 'scientific', 'academic']
    if any(kw in question.lower() for kw in research_keywords):
        return EnhancedRouteQuery(
            datasource="hybrid_web_research",
            confidence=0.85,
            reasoning="Research-oriented query",
            query_complexity="complex",
            temporal_context=False
        )
    
    # Rule 6: General knowledge (low ambiguity) → direct_llm
    if intent.ambiguity_score < 0.3 and not intent.needs_documents and not intent.needs_retrieval:
        return EnhancedRouteQuery(
            datasource="direct_llm",
            confidence=0.8,
            reasoning="Clear general knowledge question",
            query_complexity="simple",
            temporal_context=False
        )
    
    # Rule 7: Conversational queries → direct_llm
    if intent.intent_type == "conversational":
        return EnhancedRouteQuery(
            datasource="direct_llm",
            confidence=0.9,
            reasoning="Conversational query",
            query_complexity="simple",
            temporal_context=False
        )
    
    # FALLBACK: Use LLM routing only for ambiguous cases
    try:
        logger.info("Using LLM routing for ambiguous case")
        
        # SIMPLIFIED ROUTING PROMPT
        prompt = ChatPromptTemplate.from_template("""
Route this query to ONE source:
- vectorstore: User has documents uploaded
- web_search: Needs current/online info
- direct_llm: General knowledge
- research: Academic papers
- hybrid: Documents + web
- hybrid_web_research: Research + web

Query: {question}
Has documents: {has_documents}
Intent type: {intent_type}

Choose the BEST single route and explain briefly why.""")
        
        groq_service = get_groq_service(settings.groq_api_key)
        
        try:
            structured_llm = groq_service.get_structured_llm(
                settings.routing_model,
                EnhancedRouteQuery
            )
            chain = prompt | structured_llm
            
            result = groq_service.invoke_with_fallback(
                chain,
                {
                    "question": question,
                    "has_documents": has_documents,
                    "intent_type": intent.intent_type
                },
                schema=EnhancedRouteQuery
            )
            
            # Validate route
            valid_routes = ["vectorstore", "web_search", "hybrid", "direct_llm", "research", "hybrid_research", "hybrid_web_research"]
            if result.datasource not in valid_routes:
                raise ValueError(f"Invalid route: {result.datasource}")
            
            return result
            
        except Exception as llm_error:
            logger.error(f"LLM routing failed: {llm_error}, using intelligent fallback")
            
            # INTELLIGENT FALLBACK LOGIC
            if has_documents:
                return EnhancedRouteQuery(
                    datasource="hybrid",
                    confidence=0.6,
                    reasoning="Fallback: has documents, using hybrid search",
                    query_complexity="moderate",
                    temporal_context=False
                )
            else:
                return EnhancedRouteQuery(
                    datasource="web_search",
                    confidence=0.6,
                    reasoning="Fallback: no documents, using web search",
                    query_complexity="moderate",
                    temporal_context=True
                )
    
    except Exception as e:
        logger.error(f"All routing attempts failed: {e}, using safe fallback")
        
        # ULTIMATE SAFE FALLBACK
        return EnhancedRouteQuery(
            datasource="web_search" if not has_documents else "hybrid",
            confidence=0.5,
            reasoning="Emergency fallback due to routing failures",
            query_complexity="unknown",
            temporal_context=False
        )

def route_question(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    PRODUCTION-GRADE: Multi-tier routing with RAG awareness.
    
    Architecture:
    - Tier 0: Working Memory (0-2ms)
    - Tier 1: Rule-Based (2-10ms)  
    - Tier 2: Semantic (10-50ms)
    - Tier 3: RAG-Aware (50-200ms)
    
    Research: RAGRouter (2025), EMNLP 2024, Semantic Router
    """
    logger.info("NODE: route_question (PRODUCTION 4-TIER ROUTING)")
    
    question = state["question"]
    session_id = state["session_id"]
    history = state.get("conversation_history", [])
    working_memory = state.get("working_memory", {})
    
    # Check document availability
    has_documents = False
    try:
        embedding_service = get_embedding_service()
        vector_store = VectorStoreService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=f"session_{session_id}",
            embedding_dim=embedding_service.get_dimension(),
            auto_create=False
        )
        collection_info = vector_store.get_collection_info()
        doc_count = collection_info.get('points_count', 0)
        has_documents = doc_count > 0
        logger.info(f"Document check: session_{session_id} has {doc_count} documents")
    except Exception as e:
        logger.warning(f"Document check failed: {e}")
    
    # Initialize advanced router
    from ..services.advanced_router import get_advanced_router
    
    router = get_advanced_router(
        embedding_service=embedding_service if has_documents else None,
        has_documents=has_documents
    )
    
    # Execute multi-tier routing
    decision = router.route(
        query=question,
        conversation_context=history,
        working_memory=working_memory,
        has_documents=has_documents
    )
    
    # Log decision with full context
    logger.info(f"""
🎯 ROUTING DECISION:
   - Route: {decision.route.value}
   - Confidence: {decision.confidence:.2%}
   - Tier: {decision.tier} (Tier 0=fastest, Tier 3=most comprehensive)
   - Latency: {decision.latency_ms:.1f}ms
   - Reasoning: {decision.reasoning}
    """)
    
    # Map RouteType to workflow route names
    route_mapping = {
        "working_memory": "direct_llm",  # Use direct_llm with working memory
        "direct_llm": "direct_llm",
        "vectorstore": "vectorstore",
        "hybrid": "hybrid",
        "web_search": "web_search",
        "research": "research",
        "hybrid_research": "hybrid_research",
        "clarification": "clarification"
    }
    
    route_decision = route_mapping.get(decision.route.value, "direct_llm")
    
    # Update state
    return {
        **state,
        "route_decision": route_decision,
        "routing_confidence": decision.confidence,
        "routing_tier": decision.tier,
        "routing_latency_ms": decision.latency_ms,
        "routing_metadata": decision.metadata
    }





def handle_clarification_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced to detect if user switched context during clarification.
    """
    logger.info("NODE: handle_clarification_response")
    
    user_response = state["question"]
    clarification_type = state.get("clarification_type", "")
    original_question = state.get("original_question", "")
    session_id = state["session_id"]
    
    # Detect if user completely changed topic
    context_switch = detect_context_switch(user_response, [
        {"role": "assistant", "content": state.get("clarification_message", "")},
        {"role": "user", "content": original_question}
    ])
    
    if context_switch["switch_detected"] and context_switch["confidence"] > 0.8:
        # User switched to new topic - treat as new query
        logger.info("User switched context during clarification - treating as new query")
        state["dialog_state"] = "normal"
        state["needs_clarification"] = False
        # Recursively route the new question
        return route_question(state)
    
    # Analyze user's response to clarification
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the user's response to a clarification request.

Original question: {original_question}
Clarification type: {clarification_type}
User response: {user_response}

Determine the user's intent:
1. proceed_web_search - User wants to search the web
2. wait_for_upload - User will upload documents
3. provide_context - User provided missing context
4. change_query - User changed their question entirely
5. cancel - User wants to cancel/skip

Be flexible in interpretation - users may not respond exactly as expected."""),
        ("human", "What is the user's intent?")
    ])
    
    groq_service = get_groq_service(settings.groq_api_key)
    llm = groq_service.get_llm(settings.routing_model, temperature=0)
    chain = prompt | llm | StrOutputParser()
    
    response_intent = chain.invoke({
        "original_question": original_question,
        "clarification_type": clarification_type,
        "user_response": user_response
    })
    
    # Route based on user's response
    if "web" in response_intent.lower() or "search" in user_response.lower():
        logger.info("User chose web search after clarification")
        return {
            **state,
            "route_decision": "web_search",
            "question": original_question,
            "dialog_state": "clarified",
            "needs_clarification": False
        }
    
    elif "upload" in response_intent.lower() or "wait" in user_response.lower():
        logger.info("User will upload documents")
        return {
            **state,
            "route_decision": "wait_for_upload",
            "generation": "I'll wait for you to upload the documents. Please upload them using the panel on the left, then ask your question again.",
            "dialog_state": "normal",
            "needs_clarification": False
        }
    
    elif "context" in response_intent.lower() or "change_query" in response_intent.lower():
        # User provided new context or changed question - re-route
        logger.info("User provided new context/question - re-routing")
        state["dialog_state"] = "normal"
        state["needs_clarification"] = False
        return route_question(state)
    
    else:
        # Default: proceed with best guess
        logger.info("Proceeding with best guess after unclear clarification response")
        has_documents = check_document_availability(session_id)
        return {
            **state,
            "route_decision": "web_search" if not has_documents else "vectorstore",
            "question": original_question,
            "dialog_state": "normal",
            "needs_clarification": False
        }


# ========== RETRIEVAL NODES (UNCHANGED) ==========


def retrieve_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ENHANCED: Retrieve documents with query rewriting, hybrid search, and reranking.
    
    Pipeline:
    1. Query rewriting (if enabled)
    2. Hybrid search: BM25 + semantic retrieval
    3. Reranking with cross-encoder
    4. Return top-K documents
    
    Pattern: Advanced RAG pipeline (EMNLP 2024 best practices)
    """
    logger.info("NODE: retrieve_documents (ENHANCED with reranking & hybrid search)")
    
    question = state.get("enriched_question", state["question"])
    original_question = state["question"]
    session_id = state["session_id"]
    route_decision = state.get("route_decision", "vectorstore")
    
    try:
        # ========== STEP 1: Query Rewriting ==========
        rewritten_query = question
        if settings.enable_query_rewriting and route_decision in ["vectorstore", "hybrid"]:
            try:
                query_rewriter = get_query_rewriter()
                if query_rewriter.should_rewrite(question, route_decision):
                    rewrite_result = query_rewriter.rewrite_query(question)
                    rewritten_query = rewrite_result["rewritten_query"]
                    logger.info(f"✅ Query rewritten: '{question}' → '{rewritten_query}'")
                    state["rewritten_question"] = rewritten_query
                    state["query_rewritten"] = True
                else:
                    logger.info("Query rewriting skipped (not needed)")
                    state["query_rewritten"] = False
            except Exception as e:
                logger.warning(f"Query rewriting failed: {e}, using original")
                state["query_rewritten"] = False
        else:
            state["query_rewritten"] = False
        
        # Use rewritten query for retrieval
        retrieval_query = rewritten_query
        
        # ========== STEP 2: Get Embedding Service ==========
        embedding_service = get_embedding_service()
        
        # ========== STEP 3: Create Vector Store Connection ==========
        vector_store = VectorStoreService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=f"session_{session_id}",
            embedding_dim=embedding_service.get_dimension(),
            auto_create=False
        )
        
        # ========== STEP 4: Semantic Search (always performed) ==========
        logger.info(f"Performing semantic search for: '{retrieval_query}'")
        query_embedding = embedding_service.embed_text(retrieval_query)
        
        semantic_results = vector_store.similarity_search(
            query_embedding=query_embedding,
            k=settings.retrieval_k  # Retrieve more for reranking (default: 20)
        )
        
        # Convert to LangChain Documents
        semantic_documents = [
            Document(
                page_content=result["text"],
                metadata={
                    **result["metadata"],
                    "similarity_score": result.get("score", 0),
                    "retrieval_method": "semantic"
                }
            )
            for result in semantic_results
        ]
        
        logger.info(f"Semantic search retrieved {len(semantic_documents)} documents")
        
        # Store initial retrieval
        state["initial_documents"] = semantic_documents
        
        # ========== STEP 5: Hybrid Search (BM25 + Semantic) ==========
        final_documents = semantic_documents
        
        if settings.enable_hybrid_search:
            try:
                logger.info("Applying hybrid search (BM25 + semantic)")
                
                # Get all documents for BM25 indexing
                all_docs_for_bm25 = vector_store.get_all_documents_for_session()
                
                if all_docs_for_bm25:
                    # Convert to LangChain Document format for BM25
                    bm25_corpus_docs = [
                        Document(
                            page_content=doc["text"],
                            metadata=doc["metadata"]
                        )
                        for doc in all_docs_for_bm25
                    ]
                    
                    # Build BM25 index
                    hybrid_search = get_hybrid_search()
                    hybrid_search.build_bm25_index(bm25_corpus_docs)
                    
                    # Perform hybrid search
                    hybrid_documents = hybrid_search.hybrid_search(
                        query=retrieval_query,
                        semantic_results=semantic_documents,
                        top_k=settings.retrieval_k
                    )
                    
                    final_documents = hybrid_documents
                    logger.info(f"✅ Hybrid search merged results: {len(final_documents)} documents")
                else:
                    logger.warning("No documents available for BM25 indexing")
                    
            except Exception as e:
                logger.error(f"Hybrid search failed: {e}, using semantic only")
                final_documents = semantic_documents
        
        # ========== STEP 6: Reranking ==========
        if settings.enable_reranking and len(final_documents) > 0:
            try:
                logger.info(f"Reranking {len(final_documents)} documents...")
                
                reranker = get_reranker()
                reranked_documents = reranker.rerank_documents(
                    query=retrieval_query,
                    documents=final_documents,
                    top_k=settings.retrieval_after_rerank  # Default: 5
                )
                
                logger.info(f"✅ Reranking complete: kept top {len(reranked_documents)} documents")
                state["reranked_documents"] = reranked_documents
                state["reranking_applied"] = True
                final_documents = reranked_documents
                
            except Exception as e:
                logger.error(f"Reranking failed: {e}, using original order")
                state["reranking_applied"] = False
        else:
            state["reranking_applied"] = False
            # If reranking disabled, just take top-K
            final_documents = final_documents[:settings.retrieval_after_rerank]
        
        # ========== STEP 7: Return Results ==========
        logger.info(f"✅ Document retrieval complete: {len(final_documents)} documents")
        
        return {
            **state,
            "documents": final_documents
        }
        
    except Exception as e:
        logger.error(f"Document retrieval error: {e}", exc_info=True)
        return {
            **state,
            "documents": [],
            "web_search_needed": True  # Fallback to web search
        }


def grade_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Grade document relevance with RELAXED fallback threshold.
    
    Pattern: CRAG (Corrective RAG) document grading
    Critical Fix: Changed fallback threshold from 50% to 20% (only fallback if <20% relevant)
    Source: Corrective RAG paper (2024), EMNLP Best Practices
    """
    logger.info("NODE: grade_documents (FIXED with relaxed threshold)")
    
    question = state.get("rewritten_question", state.get("enriched_question", state["question"]))
    documents = state.get("documents", [])
    
    if not documents:
        logger.warning("No documents to grade")
        return {
            **state,
            "documents": [],
            "total_retrieved": 0,
            "relevant_count": 0,
            "relevance_ratio": 0.0,
            "web_search_needed": False  # CHANGED: Don't trigger fallback if no docs retrieved
        }
    
    # Initialize grading service
    groq_service = get_groq_service(settings.groq_api_key)
    
    # Grading prompt
    grade_prompt = ChatPromptTemplate.from_template("""You are a grading expert assessing document relevance.

**Document Content:**
{document}

**User Question:**
{question}

**Task:** Does this document contain information relevant to answering the question?

Return ONLY:
- "yes" if relevant (even partially relevant)
- "no" if completely irrelevant

Be lenient - mark as "yes" if the document provides ANY useful information for the question.""")
    
    llm = groq_service.get_llm(settings.grading_model, temperature=0)
    grader_chain = grade_prompt | llm | StrOutputParser()
    
    filtered_docs = []
    total_retrieved = len(documents)
    
    for i, doc in enumerate(documents, 1):
        try:
            # Extract document content
            if isinstance(doc, dict):
                doc_content = doc.get("page_content", str(doc))
            elif hasattr(doc, "page_content"):
                doc_content = doc.page_content
            else:
                doc_content = str(doc)
            
            # Grade the document
            grade_result = grader_chain.invoke({
                "document": doc_content[:1000],  # First 1000 chars
                "question": question
            })
            
            grade = grade_result.strip().lower()
            
            if "yes" in grade:
                filtered_docs.append(doc)
                logger.info(f"✓ Document {i}/{total_retrieved} relevant")
            else:
                logger.info(f"✗ Document {i}/{total_retrieved} not relevant")
                
        except Exception as e:
            logger.warning(f"Grading error for doc {i}: {e}, keeping by default")
            filtered_docs.append(doc)  # Default: keep if grading fails
    
    # Calculate metrics
    relevant_count = len(filtered_docs)
    relevance_ratio = relevant_count / total_retrieved if total_retrieved > 0 else 0.0
    
    # CRITICAL FIX: Relaxed fallback logic
    # OLD: web_search_needed = relevance_ratio < 0.5 (too aggressive)
    # NEW: web_search_needed = relevant_count == 0 (only if NO relevant docs)
    web_search_needed = (
        relevant_count == 0 or 
        relevance_ratio < settings.relevance_threshold  # Default: 0.2 (20%)
    )
    
    logger.info(f"""
📊 Grading Results:
   - Total retrieved: {total_retrieved}
   - Relevant: {relevant_count} ({relevance_ratio:.0%})
   - Threshold: {settings.relevance_threshold:.0%}
   - Web search needed: {web_search_needed}
    """)
    
    return {
        **state,
        "documents": filtered_docs,
        "total_retrieved": total_retrieved,
        "relevant_count": relevant_count,
        "relevance_ratio": relevance_ratio,
        "web_search_needed": web_search_needed
    }


def web_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform web search for current information.
    Pattern: Web-augmented retrieval
    """
    logger.info("NODE: web_search")
    
    question = state.get("enriched_question", state["question"])  # FIXED: Use enriched
    
    try:
        web_search_service = get_web_search_service(settings.tavily_api_key)
        results = web_search_service.search(question, max_results=settings.web_search_results)
        
        # Convert to Document format
        documents = [
            Document(
                page_content=result["content"],
                metadata={
                    "source": result["url"],
                    "title": result.get("title", "Web Result"),
                    "type": "web_search"
                }
            )
            for result in results
        ]
        
        logger.info(f"Web search returned {len(documents)} results")
        
        # Merge with existing documents if any
        existing_docs = state.get("documents", [])
        all_documents = existing_docs + documents
        
        return {
            **state,
            "documents": all_documents
        }
    
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return state


def research_search(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search academic papers using Semantic Scholar API with intelligent query rewriting.
    Pattern: Rewrite-Retrieve-Read (Microsoft Research EMNLP 2023)
    """
    logger.info("NODE: research_search (WITH QUERY REWRITING)")
    question = state["question"]
    
    try:
        # Initialize research search service WITH LLM for query rewriting
        groq_service = get_groq_service(settings.groq_api_key)
        research_service = get_research_search_service(llm_service=groq_service)
        
        # Search with query rewriting enabled (NEW)
        papers = research_service.search_papers(
            query=question,
            limit=settings.research_papers_limit,
            year_from=settings.research_year_threshold,
            min_citations=settings.research_citation_threshold,
            use_query_rewriting=True  # CRITICAL: Enable query rewriting
        )
        
        if not papers:
            logger.warning("No research papers found after query rewriting")
            return {
                **state,
                "research_papers": [],
                "documents": []
            }
        
        # Format papers as LangChain Documents
        documents = []
        research_papers = []
        
        for paper in papers:
            research_papers.append(paper)
            content = research_service.format_paper_for_context(paper)
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": "research",
                    "paper_id": paper.get("paperId"),
                    "title": paper.get("title", "Unknown"),
                    "year": paper.get("year"),
                    "citations": paper.get("citationCount", 0),
                    "url": paper.get("url"),
                    "authors": [a.get("name") for a in paper.get("authors", [])[:3]]
                }
            )
            documents.append(doc)
        
        logger.info(f"Retrieved {len(documents)} research papers")
        
        return {
            **state,
            "research_papers": research_papers,
            "documents": documents
        }
    
    except Exception as e:
        logger.error(f"Research search error: {e}")
        return {
            **state,
            "research_papers": [],
            "documents": [],
            "web_search_needed": True  # Fallback to web search
        }


def transform_query(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform the query for better retrieval.
    - If conversation history is available, it's used to add context.
    - The query is always optimized for typos, vagueness, and abbreviations.
    """
    logger.info("NODE: transform_query")

    question = state["question"]
    # Use "conversation_history" to match your existing state key
    messages = state.get("conversation_history", [])

    system_prompt = ""
    # Dynamically change the prompt based on whether history is available
    if messages:
        # --- Prompt for when CONTEXT IS AVAILABLE ---
        logger.info("Context found. Using context-aware query transformation.")
        system_prompt = """You are an expert at rewriting user queries for a Retrieval-Augmented Generation (RAG) system.
Your task is to transform a user's latest query into a standalone, clear query optimized for search.

**Instructions:**
1.  **Use Chat History for Context:** The user's query may be a follow-up. Use the provided chat history to resolve references (like "it", "that", "the previous paper").
2.  **Optimize the Query:** Expand abbreviations, fix typos, use more specific terminology, and add synonyms.
3.  **Combine Both:** Merge the context from the history with the query optimization to create one detailed, standalone query.
4.  **Do not answer the query.** Your only job is to rewrite it for a search system.
5.  Return ONLY the rewritten query as a single string.

**Chat History:**
{chat_history}"""
        
        # Format the history into a simple string for the prompt
        input_vars = {
            "question": question,
            "chat_history": "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[-8:]])
        }
        human_template = "User's Latest Query: {question}\n\nTransformed query:"
        
    else:
        # --- Your ORIGINAL Prompt for when NO CONTEXT is available ---
        logger.info("No context found. Using standard query optimization.")
        system_prompt = """You are an expert at query optimization for information retrieval.

Given a user question, generate an improved search query that:
1. Expands abbreviations and acronyms
2. Adds relevant context if obvious (e.g., from the query itself)
3. Uses more specific terminology
4. Includes synonyms or related terms

Keep the transformed query concise and focused.
Return ONLY the transformed query, nothing else."""
        
        input_vars = {"question": question}
        human_template = "Original query: {question}\n\nTransformed query:"

    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])
    
    # Set up the LLM chain
    groq_service = get_groq_service(settings.groq_api_key)
    # Use a faster model for this task to reduce latency
    llm = groq_service.get_llm(settings.routing_model, temperature=0.0) 
    chain = prompt | llm | StrOutputParser()
    
    try:
        transformed = chain.invoke(input_vars)
        
        # Avoid minor, unnecessary changes
        if transformed.strip().lower() == question.strip().lower():
            logger.info("Query transformation resulted in an identical query. No changes made.")
            return state

        logger.info(f"Query transformed: '{question}' → '{transformed}'")
        
        # Return the state with the updated question
        return { **state, "question": transformed }
    
    except Exception as e:
        logger.error(f"Query transformation failed: {e}. Using original query.")
        # On failure, always return the original state to prevent the workflow from breaking
        return state

# ========== GENERATION NODES (FIXED WITH WORKING MEMORY) ==========

def generate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    ENHANCED: Generate with document awareness, reverse repacking, and working memory.
    
    Features:
    - Document registry context
    - Filename-aware citations
    - Reverse document repacking (most relevant last)
    - Working memory integration
    - Conversation history
    
    Pattern: Document-aware RAG (Google AI 2024) + Context repacking (EMNLP 2024)
    """
    logger.info("NODE: generate (ENHANCED with document awareness + reverse repacking)")
    
    # Add capability and document context
    state = add_capability_context_to_state(state)
    state = add_document_context_to_state(state)  # NEW: Document awareness
    
    question = state.get("enriched_question", state["question"])
    documents = state.get("documents", [])
    history = state.get("conversation_history", [])
    working_memory = state.get("working_memory", {})
    capabilities = state.get("system_capabilities", "")
    document_context = state.get("document_context", "")  # NEW
    
    # ENHANCED: Format documents with FILENAME + reverse repacking
    if documents:
        # Reverse order so most relevant is LAST (LLM recency bias)
        documents_for_generation = list(reversed(documents[:5]))
        
        logger.info(f"📄 Repacking {len(documents_for_generation)} documents (most relevant LAST)")
        
        # Group by filename for clear attribution
        docs_by_file = {}
        for doc in documents_for_generation:
            filename = doc.metadata.get("filename", "Unknown Document")
            if filename not in docs_by_file:
                docs_by_file[filename] = []
            docs_by_file[filename].append(doc)
        
        # Format with filename headers
        context_parts = []
        for filename, file_docs in docs_by_file.items():
            context_parts.append(f"\n{'='*60}")
            context_parts.append(f"📄 SOURCE: **{filename}**")
            context_parts.append(f"{'='*60}\n")
            
            for doc in file_docs:
                chunk_idx = doc.metadata.get("chunk_index", "?")
                page = doc.metadata.get("page", "")
                page_info = f" (Page {page})" if page else ""
                
                context_parts.append(f"[Chunk {chunk_idx}{page_info}]\n{doc.page_content}\n")
        
        context = "\n".join(context_parts)
        
        # Log repacking order
        for i, doc in enumerate(documents_for_generation):
            rerank_score = doc.metadata.get('rerank_score', 'N/A')
            filename = doc.metadata.get('filename', 'Unknown')
            position = "LEAST relevant" if i == 0 else ("MOST relevant" if i == len(documents_for_generation)-1 else "moderate")
            logger.debug(f"  Position {i+1}: {filename} - {position} (score={rerank_score})")
    else:
        context = "No specific documents or sources available."
    
    # Format conversation history
    conversation_context = ""
    if history:
        recent_history = history[-6:]
        for msg in recent_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            conversation_context += f"{role}: {msg.get('content', '')}\n"
    
    # Format working memory
    memory_text = format_working_memory_context(working_memory)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """{capabilities}

{document_context}

{memory_context}

**CRITICAL RULES FOR DOCUMENT REFERENCES:**
1. **ALWAYS use actual filenames** when referencing documents (e.g., "analysis.pdf", "report_2024.docx")
2. **NEVER use generic terms** like "the document", "the first file", "the file with 84 chunks"
3. When multiple documents exist, specify which one: "According to **analysis.pdf**..."
4. Use natural citations: "In **report.pdf** (page 3), the findings show..."

**Examples of GOOD references:**
✓ "According to **analysis.pdf**, the main findings are..."
✓ "**report_2024.docx** indicates that..."
✓ "Comparing **data.csv** with **results.xlsx**..."

**Examples of BAD references (NEVER DO THIS):**
✗ "The first file shows..."
✗ "In the document with 84 chunks..."
✗ "The PDF you uploaded..."
✗ "According to the document..."

**Response Guidelines:**
- Synthesize information from retrieved documents and web search results
- **Cite document filenames naturally** throughout your response
- Use working memory for computational continuity
- Be concise but comprehensive
- For computational queries, show your work step-by-step
- If you don't have specific information, state what you found

**IMPORTANT: Documents are ordered with MOST RELEVANT at the END** - pay attention to later chunks."""),
        ("human", """Retrieved Context from Documents:
{context}

Conversation History:
{conversation_history}

Current Question: {question}

Answer:""")
    ])
    
    groq_service = get_groq_service(settings.groq_api_key)
    llm = groq_service.get_llm(settings.generation_model, temperature=0.7)
    chain = prompt | llm | StrOutputParser()
    
    try:
        generation = chain.invoke({
            "capabilities": capabilities,
            "document_context": document_context,  # NEW: Document list
            "memory_context": memory_text,
            "context": context,
            "conversation_history": conversation_context,
            "question": question
        })
        
        # Extract sources with filenames
        sources = []
        for doc in documents[:5]:  # Original top-5 order
            filename = doc.metadata.get("filename", "Unknown")
            source_info = {
                "url": doc.metadata.get("source", ""),
                "title": filename,  # Use filename as title
                "type": doc.metadata.get("type", "document"),
                "filename": filename  # NEW: Explicit filename field
            }
            if source_info not in sources:
                sources.append(source_info)
        
        logger.info(f"✅ Generated answer using {len(sources)} sources with document awareness")
        
        return {
            **state,
            "generation": generation,
            "sources": sources
        }
    
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return {
            **state,
            "generation": "I apologize, but I encountered an error generating the response. Please try again.",
            "sources": []
        }


def add_document_context_to_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add document registry context to state.
    
    Provides LLM with list of uploaded documents and their metadata.
    Pattern: Document-aware dialog (Google AI 2024)
    """
    session_id = state.get("session_id")
    if not session_id:
        return state
    
    try:
        from ..services.document_registry import get_document_registry
        doc_registry = get_document_registry()
        
        # Get document summary
        document_summary = doc_registry.get_document_summary(session_id)
        session_docs = doc_registry.get_session_documents(session_id)
        
        # Format for LLM
        if session_docs:
            context = f"""
📚 **DOCUMENTS AVAILABLE IN THIS SESSION:**
{document_summary}

**CRITICAL:** When referencing these documents, ALWAYS use their exact filenames.
For example: "According to **{session_docs[0]['filename']}**..." NOT "According to the document..."
"""
            state["document_context"] = context
            state["session_documents"] = session_docs
            
            logger.debug(f"✅ Added document context: {len(session_docs)} documents")
        else:
            state["document_context"] = ""
            state["session_documents"] = []
        
    except Exception as e:
        logger.warning(f"Failed to add document context: {e}")
        state["document_context"] = ""
        state["session_documents"] = []
    
    return state




def direct_llm_generate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Generate answer directly from LLM without retrieval, with working memory support.
    Pattern: Direct generation for general knowledge queries
    """
    logger.info("NODE: direct_llm_generate (FIXED with memory)")
    
    # Add capability context to state
    state = add_capability_context_to_state(state)
    
    question = state.get("enriched_question", state["question"])  # FIXED: Use enriched
    history = state.get("conversation_history", [])
    working_memory = state.get("working_memory", {})  # FIXED: Extract working memory
    capabilities = state.get("system_capabilities", "")
    
    # Format conversation history
    conversation_context = ""
    if history:
        recent_history = history[-6:]
        for msg in recent_history:
            role = "User" if msg.get("role") == "user" else "Assistant"
            conversation_context += f"{role}: {msg.get('content', '')}\n"
        conversation_context = f"Conversation History:\n{conversation_context}\n"
    
    # FIXED: Format working memory
    memory_text = format_working_memory_context(working_memory)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with multiple capabilities.

{capabilities}

{memory_context}

**CRITICAL RULES:**
1. Never say you cannot search the web - you have that capability through the system
2. Never say you cannot access documents - the system routes to appropriate sources
3. Never apologize for limitations you do not have - web search and document access are available
4. For computational queries: CHECK the working memory for stored variables
5. Use stored variable values in your calculations

**Computational Example:**
If working memory shows c = 3 and user asks "c+1", answer "4"

**Response Guidelines:**
- For computational queries: Solve step-by-step and show your work
- For general knowledge: Provide accurate, concise information
- For conversational follow-ups: Use conversation history and working memory
- Never claim you lack capabilities that the system provides"""),
        ("human", """{conversation_context}

Current Question: {question}

Answer:""")
    ])
    
    groq_service = get_groq_service(settings.groq_api_key)
    llm = groq_service.get_llm(settings.generation_model, temperature=0.7)
    chain = prompt | llm | StrOutputParser()
    
    try:
        generation = chain.invoke({
            "capabilities": capabilities,
            "memory_context": memory_text,  # FIXED: Inject memory
            "conversation_context": conversation_context,
            "question": question
        })
        
        return {
            **state,
            "generation": generation,
            "sources": []
        }
    
    except Exception as e:
        logger.error(f"Direct LLM generation error: {e}")
        return {
            **state,
            "generation": "I apologize, but I encountered an error. Please try again.",
            "sources": []
        }


def generate_clarification(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    FIXED: Generate clarification message for user and ensure 'generation' field is set.
    Pattern: Proactive clarification dialog
    """
    logger.info("NODE: generate_clarification (FIXED)")
    
    clarification_message = state.get("clarification_message", "")
    clarification_options = state.get("clarification_options", [])
    
    # Format options if provided
    if clarification_options:
        options_text = "\n\nOptions:\n" + "\n".join([
            f"- {option}" for option in clarification_options
        ])
        full_message = clarification_message + options_text
    else:
        full_message = clarification_message
    
    # CRITICAL FIX: Set 'generation' field so response is sent to user
    return {
        **state,
        "generation": full_message,
        "sources": []
    }


def hybrid_generate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate using both retrieved documents and web search.
    Pattern: Hybrid RAG for comprehensive answers
    """
    logger.info("NODE: hybrid_generate")
    
    # First perform web search to augment documents
    state = web_search(state)
    
    # Then generate with all available context
    return generate(state)


def hybrid_web_research_generate(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    PRODUCTION-GRADE: Combined research papers + web search + document retrieval.
    
    Pattern: Multi-source RAG with intelligent query rewriting
    Source: Rewrite-Retrieve-Read (Microsoft EMNLP 2023), Advanced RAG 2025
    
    Features:
    - LLM-powered query rewriting for research papers (15-20% improvement)
    - Parallel multi-source retrieval (documents + research + web)
    - Citation-aware ranking
    - Fallback handling at each step
    """
    logger.info("NODE: hybrid_web_research_generate")
    
    question = state["question"]
    session_id = state["session_id"]
    
    # Storage for all retrieved sources
    all_documents = []
    research_papers_list = []
    web_results_list = []
    
    # ========== PART 1: RESEARCH PAPERS WITH QUERY REWRITING (CRITICAL FIX) ==========
    try:
        logger.info("STEP 1: Searching research papers with query rewriting...")
        
        # Initialize services
        groq_service = get_groq_service(settings.groq_api_key)
        research_service = get_research_search_service(
            api_key=settings.semantic_scholar_api_key,
            llm_service=groq_service  # CRITICAL: Enable LLM query rewriting
        )
        
        # CRITICAL: Search with query rewriting enabled
        research_papers = research_service.search_papers(
            query=question,
            limit=settings.research_papers_limit,
            year_from=settings.research_year_threshold,
            min_citations=settings.research_citation_threshold,
            use_query_rewriting=True  # NEW: Enable intelligent query rewriting
        )
        
        logger.info(f"Retrieved {len(research_papers)} research papers")
        
        # Format papers as LangChain Documents
        for paper in research_papers:
            research_papers_list.append(paper)
            
            # Format paper into structured context
            content = research_service.format_paper_for_context(paper)
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": "semantic_scholar",
                    "type": "research_paper",
                    "paper_id": paper.get("paperId"),
                    "title": paper.get("title", "Unknown"),
                    "year": paper.get("year"),
                    "citations": paper.get("citationCount", 0),
                    "venue": paper.get("venue", ""),
                    "url": paper.get("url", ""),
                    "authors": [a.get("name") for a in paper.get("authors", [])[:3]]
                }
            )
            all_documents.append(doc)
    
    except Exception as e:
        logger.error(f"Research search error: {e}", exc_info=True)
        # Continue to web search even if research fails
    
    # ========== PART 2: WEB SEARCH ==========
    try:
        logger.info("STEP 2: Performing web search...")
        
        web_search_service = get_web_search_service(settings.tavily_api_key)
        web_results = web_search_service.search(
            question,
            max_results=settings.web_search_results
        )
        
        logger.info(f"Retrieved {len(web_results)} web results")
        
        # Format web results as Documents
        for result in web_results:
            web_results_list.append(result)
            
            doc = Document(
                page_content=result.get("content", result.get("snippet", "")),
                metadata={
                    "source": "web_search",
                    "type": "web_result",
                    "title": result.get("title", "Web Result"),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0.0)
                }
            )
            all_documents.append(doc)
    
    except Exception as e:
        logger.error(f"Web search error: {e}", exc_info=True)
        # Continue even if web search fails
    
    # ========== PART 3: DOCUMENT RETRIEVAL (if documents exist) - CRITICAL FIX ==========
    try:
        logger.info("STEP 3: Retrieving from uploaded documents...")
        
        embedding_service = get_embedding_service()
        vector_store = VectorStoreService(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=f"session_{session_id}",
            embedding_dim=embedding_service.get_dimension()
        )
        
        # Check if collection has documents
        collection_info = vector_store.get_collection_info()
        if collection_info.get("points_count", 0) > 0:
            # CRITICAL FIX: Convert query to embedding BEFORE passing to similarity_search
            query_embedding = embedding_service.embed_text(question)
            
            # Now pass embedding vector (not string)
            search_results = vector_store.similarity_search(
                query_embedding=query_embedding,  # ✓ FIXED: Pass embedding vector
                k=settings.retrieval_k
            )
            
            logger.info(f"Retrieved {len(search_results)} documents from vector store")
            
            # Convert search results to LangChain Documents
            for result in search_results:
                doc = Document(
                    page_content=result["text"],
                    metadata={
                        **result["metadata"],
                        "type": "uploaded_document",
                        "score": result["score"]
                    }
                )
                all_documents.append(doc)
        else:
            logger.info("No documents in vector store for this session")
    
    except Exception as e:
        logger.error(f"Document retrieval error: {e}", exc_info=True)
        # Continue even if document retrieval fails
    
    # ========== PART 4: CHECK IF ANY SOURCES RETRIEVED ==========
    if not all_documents:
        logger.warning("No sources retrieved from any channel (research/web/docs)")
        
        # Fallback to direct LLM generation
        return _fallback_generation(state, question, 
                                      "No research papers, web results, or documents found.")
    
    # ========== PART 5: BUILD STRUCTURED CONTEXT ==========
    logger.info(f"Building context from {len(all_documents)} total sources")
    
    # Organize context by source type
    context_parts = []
    
    # 5.1: Research Papers Section
    research_docs = [d for d in all_documents if d.metadata.get("type") == "research_paper"]
    if research_docs:
        research_context = "**📚 Recent Research Papers:**\n\n"
        for i, doc in enumerate(research_docs, 1):
            meta = doc.metadata
            research_context += f"{i}. **{meta.get('title', 'Unknown')}**\n"
            research_context += f"   Authors: {', '.join(meta.get('authors', ['Unknown']))}\n"
            research_context += f"   Year: {meta.get('year', 'N/A')} | Citations: {meta.get('citations', 0)} | Venue: {meta.get('venue', 'N/A')}\n"
            research_context += f"   {doc.page_content[:400]}...\n\n"
        
        context_parts.append(research_context)
    
    # 5.2: Web Results Section
    web_docs = [d for d in all_documents if d.metadata.get("type") == "web_result"]
    if web_docs:
        web_context = "**🌐 Web Sources:**\n\n"
        for i, doc in enumerate(web_docs, 1):
            meta = doc.metadata
            web_context += f"{i}. **{meta.get('title', 'Web Result')}**\n"
            web_context += f"   {doc.page_content[:300]}...\n"
            if meta.get('url'):
                web_context += f"   Source: {meta['url']}\n\n"
        
        context_parts.append(web_context)
    
    # 5.3: Uploaded Documents Section
    doc_docs = [d for d in all_documents if d.metadata.get("type") == "uploaded_document"]
    if doc_docs:
        doc_context = "**📄 From Uploaded Documents:**\n\n"
        for i, doc in enumerate(doc_docs, 1):
            doc_context += f"{i}. {doc.page_content[:300]}...\n\n"
        
        context_parts.append(doc_context)
    
    combined_context = "\n\n".join(context_parts)
    
    # ========== PART 6: GENERATE COMPREHENSIVE RESPONSE ==========
    try:
        logger.info("STEP 4: Generating response with multi-source context...")
        
        prompt = ChatPromptTemplate.from_template("""You are an expert research assistant providing comprehensive answers by synthesizing information from multiple sources.

**Question:** {question}

**Context from Multiple Sources:**
{context}

**Instructions:**
1. **Synthesize information** from ALL sources (research papers, web, documents)
2. **Prioritize peer-reviewed research** when available, but integrate web and document sources
3. **Cite specific sources**:
   - For research papers: Mention authors, year, and key findings
   - For web sources: Reference the source title
   - For documents: Quote relevant sections
4. **Establish credibility**: Mention citation counts, venues, publication dates
5. **Highlight trends/timeline** if query asks about recent developments
6. **Compare perspectives** if multiple sources provide different views
7. **Be critical**: Note if sources conflict or if information is limited
8. If comparing uploaded documents with research, clearly distinguish between the two

**Answer Format:**
- Start with a direct answer (2-3 sentences)
- Organize findings by theme or chronology
- Use specific citations and examples
- End with implications or future directions if relevant

Provide a comprehensive, well-cited answer:""")
        
        groq_service = get_groq_service(settings.groq_api_key)
        llm = groq_service.get_llm(settings.generation_model, temperature=0.3)
        chain = prompt | llm | StrOutputParser()
        
        generation = chain.invoke({
            "question": question,
            "context": combined_context
        })
        
        logger.info(f"✅ Hybrid web research generated response with {len(all_documents)} total sources")
        
        return {
            **state,
            "generation": generation,
            "documents": all_documents,  # All sources combined
            "research_papers": research_papers_list,  # Original paper dicts
            "web_results": web_results_list  # Original web result dicts
        }
    
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        
        # Fallback to simpler generation
        return _fallback_generation(
            state, 
            question, 
            f"Generation error: {str(e)}\n\nSources found: {len(all_documents)}"
        )


def _fallback_generation(state: Dict[str, Any], question: str, error_msg: str) -> Dict[str, Any]:
    """
    Fallback generation when hybrid search fails.
    Pattern: Graceful degradation with transparency
    """
    logger.warning(f"Using fallback generation: {error_msg}")
    
    try:
        fallback_prompt = ChatPromptTemplate.from_template("""You are a helpful research assistant. 

⚠️ **Note:** Research paper and web search are temporarily unavailable.

Answer this question based on your general knowledge, but make it clear that you don't have access to the latest research papers or current web information:

**Question:** {question}

Provide a helpful answer while being transparent about limitations:""")
        
        groq_service = get_groq_service(settings.groq_api_key)
        llm = groq_service.get_llm(
            settings.generation_model,  # Use generation model (fallback_model doesn't exist in config)
            temperature=0.3
        )
        chain = fallback_prompt | llm | StrOutputParser()
        
        fallback_generation = chain.invoke({"question": question})
        
        return {
            **state,
            "generation": f"⚠️ **Research sources temporarily unavailable**\n\n{fallback_generation}\n\n---\n\n*Error details: {error_msg}*",
            "documents": [],
            "research_papers": [],
            "web_results": []
        }
    
    except Exception as fallback_error:
        logger.error(f"Fallback generation also failed: {fallback_error}")
        
        # Ultimate fallback
        return {
            **state,
            "generation": f"I encountered an error while searching for information. Please try rephrasing your question or try again later.\n\nError: {error_msg}",
            "documents": [],
            "research_papers": [],
            "web_results": []
        }

def wait_for_upload(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle waiting for document upload.
    """
    logger.info("NODE: wait_for_upload")
    
    return {
        **state,
        "generation": "Please upload your documents using the upload panel on the left, then ask your question again.",
        "sources": []
    }
