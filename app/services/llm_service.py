"""
LLM Service with Production-Ready Error Handling and Retry Logic

Pattern: Exponential Backoff with Circuit Breaker
Source: OpenAI Best Practices, LangChain Production Patterns
"""

from typing import Any, Type, Optional, Callable
from langchain_groq import ChatGroq
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
import time
import random
import json
from functools import wraps

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class CircuitBreaker:
    """Circuit breaker pattern to prevent cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - too many failures")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED")
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e

def with_retry_and_fallback(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    exponential_base: float = 2.0,
    jitter: bool = True
):
    """
    Decorator for exponential backoff with jitter on LLM calls.
    Pattern: Exponential Backoff (Industry Standard)
    Source: AWS Architecture, Google SRE Handbook
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    error_str = str(e).lower()
                    last_exception = e
                    
                    # Don't retry on validation errors (client errors)
                    if any(code in error_str for code in ['400', 'invalid', 'validation']):
                        # For Groq tool_use_failed, try fallback
                        if 'tool_use_failed' in error_str:
                            logger.warning(f"Groq tool_use_failed on attempt {attempt + 1}, will retry with simplified prompt")
                        else:
                            logger.error(f"Non-retryable error: {e}")
                            raise e
                    
                    # Retry on server errors, rate limits, timeouts
                    if attempt < max_retries - 1:
                        # Calculate exponential backoff
                        delay = min(max_delay, base_delay * (exponential_base ** attempt))
                        
                        # Add jitter (random 0-1 seconds)
                        if jitter:
                            delay += random.uniform(0, 1)
                        
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed. Last error: {e}")
            
            # All retries exhausted
            raise last_exception
        
        return wrapper
    return decorator

class GroqService:
    """Enhanced Groq Service with error handling and circuit breaker."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)
    
    def get_llm(self, model: str, temperature: float = 0.7) -> BaseChatModel:
        """Get standard LLM instance."""
        return ChatGroq(
            api_key=self.api_key,
            model=model,
            temperature=temperature,
            timeout=30.0,
            max_retries=2
        )
    
    @with_retry_and_fallback(max_retries=3, base_delay=1.0)
    def get_structured_llm(
        self,
        model: str,
        schema: Type[BaseModel],
        temperature: float = 0.0
    ) -> Any:
        """
        Get structured LLM with error handling and fallback.
        CRITICAL FIX: Wraps structured output with retry and fallback logic
        """
        try:
            llm = ChatGroq(
                api_key=self.api_key,
                model=model,
                temperature=temperature,
                timeout=30.0,
                max_retries=0
            )
            
            def _create_structured():
                return llm.with_structured_output(schema, method="function_calling")
            
            return self.circuit_breaker.call(_create_structured)
        
        except Exception as e:
            logger.error(f"Failed to create structured LLM: {e}")
            raise
    
    @with_retry_and_fallback(max_retries=3, base_delay=1.0)
    def invoke_with_fallback(
        self,
        llm_chain: Any,
        input_data: dict,
        schema: Optional[Type[BaseModel]] = None
    ) -> Any:
        """
        Invoke LLM chain with structured output fallback.
        Pattern: Graceful Degradation
        
        CRITICAL FIX: Properly escape JSON schema for ChatPromptTemplate
        """
        try:
            # Primary: Structured output
            return llm_chain.invoke(input_data)
        
        except Exception as e:
            error_str = str(e).lower()
            
            # If tool_use_failed, try fallback to unstructured + JSON parsing
            if 'tool_use_failed' in error_str and schema:
                logger.warning("Structured output failed, attempting JSON fallback")
                
                try:
                    from langchain_core.output_parsers import JsonOutputParser
                    from langchain_core.prompts import ChatPromptTemplate
                    
                    # Get unstructured LLM
                    unstructured_llm = self.get_llm(model="llama-3.1-8b-instant", temperature=0)
                    json_parser = JsonOutputParser(pydantic_object=schema)
                    
                    # CRITICAL FIX: Get JSON schema and serialize it safely
                    schema_dict = schema.model_json_schema()
                    
                    # Convert to JSON string and escape for template
                    schema_json_str = json.dumps(schema_dict, indent=2)
                    
                    # CRITICAL: Escape curly braces for LangChain template
                    # Single { becomes {{, single } becomes }}
                    escaped_schema = schema_json_str.replace("{", "{{").replace("}", "}}")
                    
                    # Build simple prompt with escaped schema
                    simple_prompt = ChatPromptTemplate.from_messages([
                        ("system", f"""You are a helpful assistant that returns valid JSON.

Return JSON matching this schema:
{escaped_schema}

IMPORTANT: Return ONLY the JSON object, no other text."""),
                        ("human", "{input}")
                    ])
                    
                    # Extract original input
                    original_input = (
                        input_data.get("question") or 
                        input_data.get("query") or 
                        input_data.get("input") or 
                        str(input_data)
                    )
                    
                    # Invoke fallback chain
                    fallback_chain = simple_prompt | unstructured_llm | json_parser
                    result = fallback_chain.invoke({"input": original_input})
                    
                    # Validate and return
                    return schema(**result)
                
                except Exception as fallback_error:
                    logger.error(f"JSON fallback also failed: {fallback_error}")
                    raise e  # Raise original error
            
            raise e

# Singleton instance
_groq_service: Optional[GroqService] = None

def get_groq_service(api_key: str) -> GroqService:
    """Get or create Groq service singleton."""
    global _groq_service
    if _groq_service is None:
        _groq_service = GroqService(api_key)
    return _groq_service
