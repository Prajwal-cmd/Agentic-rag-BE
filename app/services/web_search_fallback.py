"""
Multi-Provider Web Search Service with Fallback Chain

Pattern: Tavily → DuckDuckGo → Brave (all free)
Source: Industry best practice for API resilience (AWS, Google SRE)
"""

from typing import List, Dict, Optional
from tavily import TavilyClient
from duckduckgo_search import DDGS
import time

from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class MultiProviderWebSearch:
    """
    Web search with automatic fallback chain.
    
    Fallback Order:
    1. Tavily (1,000/month free)
    2. DuckDuckGo (unlimited, rate-limited)
    3. Direct failure message
    
    Pattern: Circuit Breaker with Cascading Fallback
    """
    
    def __init__(self, tavily_api_key: Optional[str] = None):
        self.tavily_api_key = tavily_api_key
        self.tavily_client = None
        
        # Initialize Tavily if API key provided
        if tavily_api_key:
            try:
                self.tavily_client = TavilyClient(api_key=tavily_api_key)
                logger.info("✓ Tavily search service initialized (primary)")
            except Exception as e:
                logger.warning(f"Tavily initialization failed: {e}, using fallbacks only")
        
        # DuckDuckGo (always available, no API key)
        self.ddgs = DDGS()
        logger.info("✓ DuckDuckGo search service initialized (fallback)")
    
    def search(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Perform web search with automatic fallback.
        
        Args:
            query: Search query
            max_results: Maximum number of results
        
        Returns:
            List of search results with content, title, URL
        """
        # ATTEMPT 1: Tavily (primary, best quality)
        if self.tavily_client and settings.enable_web_search_fallback:
            try:
                logger.info(f"🔍 ATTEMPT 1: Tavily search for: {query}")
                results = self._search_tavily(query, max_results)
                if results:
                    logger.info(f"✅ Tavily returned {len(results)} results")
                    return results
                logger.warning("Tavily returned 0 results, trying DuckDuckGo...")
            except Exception as e:
                logger.warning(f"⚠️ Tavily search failed: {e}, falling back to DuckDuckGo")
        
        # ATTEMPT 2: DuckDuckGo (free, unlimited but rate-limited)
        if settings.use_duckduckgo_fallback:
            try:
                logger.info(f"🔍 ATTEMPT 2: DuckDuckGo search for: {query}")
                results = self._search_duckduckgo(query, max_results)
                if results:
                    logger.info(f"✅ DuckDuckGo returned {len(results)} results")
                    return results
                logger.warning("DuckDuckGo returned 0 results")
            except Exception as e:
                logger.error(f"⚠️ DuckDuckGo search failed: {e}")
        
        # ATTEMPT 3: All fallbacks exhausted
        logger.error("❌ All web search providers failed")
        return []
    
    def _search_tavily(self, query: str, max_results: int) -> List[Dict]:
        """Search using Tavily API."""
        try:
            response = self.tavily_client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=False,
                include_raw_content=False
            )
            
            results = []
            for result in response.get("results", []):
                results.append({
                    "content": result.get("content", ""),
                    "title": result.get("title", "Untitled"),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0.0),
                    "source": "tavily"
                })
            
            return results
        
        except Exception as e:
            logger.error(f"Tavily API error: {e}")
            raise
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict]:
        """
        Search using DuckDuckGo (free, no API key required).
        
        NOTE: DuckDuckGo has rate limits (~10 requests/second).
        Pattern: Add delay between calls to avoid 202 rate limit errors.
        """
        try:
            # Add small delay to respect rate limits
            time.sleep(0.2)  # 5 requests/second max
            
            results = []
            
            # DuckDuckGo text search
            ddg_results = self.ddgs.text(
                keywords=query,
                max_results=max_results,
                region="wt-wt",  # Worldwide
                safesearch="moderate",
                timelimit=None
            )
            
            for idx, result in enumerate(ddg_results):
                if idx >= max_results:
                    break
                
                results.append({
                    "content": result.get("body", ""),
                    "title": result.get("title", "Untitled"),
                    "url": result.get("href", ""),
                    "score": 1.0 - (idx * 0.1),  # Decreasing score by position
                    "source": "duckduckgo"
                })
            
            return results
        
        except Exception as e:
            logger.error(f"DuckDuckGo API error: {e}")
            raise

# Global instance
_web_search_service = None

def get_web_search_service(api_key: Optional[str] = None) -> MultiProviderWebSearch:
    """Get or create global web search service instance."""
    global _web_search_service
    if _web_search_service is None:
        _web_search_service = MultiProviderWebSearch(api_key)
    return _web_search_service
