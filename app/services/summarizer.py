"""
Conversation Summarization Service
Pattern: Batch summarization for token efficiency (single API call)
Source: LangChain conversation memory patterns
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class ConversationSummarizer:
    """
    Efficient conversation history summarization.
    Reduces token usage while preserving context.
    """
    
    def __init__(self, groq_service, model_name: str):
        """
        Initialize summarizer.
        
        Args:
            groq_service: Groq service instance
            model_name: Model for summarization
        """
        self.groq_service = groq_service
        self.model_name = model_name
        logger.info("Conversation summarizer initialized")
    
    def summarize_history(
        self,
        messages: List[Dict],
        max_tokens: int = 200
    ) -> str:
        """
        Summarize conversation history into concise format.
        
        Pattern: Single LLM call for batch summarization (efficient)
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Target summary length
            
        Returns:
            Concise summary preserving key context
        """
        # Format messages for summarization
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}" for msg in messages
        ])
        
        # Summarization prompt - optimized for brevity and context preservation
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Summarize this conversation segment concisely.
Preserve:
- User's main questions and concerns
- Assistant's key points and information provided  
- Any document references or sources mentioned
- Overall topic flow and context

Keep under {max_tokens} tokens while maintaining coherence."""),
            ("human", "Conversation to summarize:\n\n{conversation}")
        ])
        
        # Create summarization chain
        llm = self.groq_service.get_llm(self.model_name, temperature=0)
        chain = prompt | llm | StrOutputParser()
        
        try:
            summary = chain.invoke({
                "conversation": conversation_text,
                "max_tokens": max_tokens
            })
            logger.info(f"Summarized {len(messages)} messages into {len(summary)} characters")
            return summary
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            # Fallback: truncate if summarization fails
            return conversation_text[:500] + "..."
    
    def should_summarize(self, messages: List[Dict], threshold: int = 10) -> bool:
        """
        Check if conversation should be summarized.
        
        Args:
            messages: Current message history
            threshold: Number of messages before summarization
            
        Returns:
            True if summarization recommended
        """
        return len(messages) > threshold
    
    def compress_history(
        self,
        messages: List[Dict],
        recent_k: int = 4
    ) -> List[Dict]:
        """
        Compress history: summarize old messages, keep recent ones verbatim.
        
        Pattern: Hybrid approach balancing context and efficiency
        
        Args:
            messages: Full message history
            recent_k: Number of recent messages to keep verbatim
            
        Returns:
            Compressed history with summary + recent messages
        """
        if len(messages) <= recent_k:
            return messages
        
        # Split into old and recent
        old_messages = messages[:-recent_k]
        recent_messages = messages[-recent_k:]
        
        # Summarize old messages
        summary = self.summarize_history(old_messages)
        
        # Create compressed history
        compressed = [
            {"role": "system", "content": f"Previous conversation summary: {summary}"}
        ] + recent_messages
        
        logger.info(f"Compressed {len(messages)} messages to {len(compressed)} (summary + {recent_k} recent)")
        return compressed