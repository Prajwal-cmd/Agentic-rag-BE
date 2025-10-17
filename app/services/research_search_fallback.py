"""
Multi-Provider Academic Search with Fallback Chain

Pattern: Semantic Scholar → arXiv → CORE (all free)
Source: Academic RAG best practices
"""

import requests
import arxiv
from typing import List, Dict, Optional, Any
import re
import time

from ..config import settings
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class MultiProviderResearchSearch:
    """
    Academic paper search with automatic fallback chain.
    
    Fallback Order:
    1. Semantic Scholar (rate-limited, ~100/day free)
    2. arXiv API (unlimited, 3 sec/call throttle)
    3. CORE API (1,000/day free)
    
    Pattern: Cascading Fallback with Query Adaptation
    """
    
    def __init__(self, semantic_scholar_api_key: Optional[str] = None, llm_service=None):
        self.semantic_scholar_api_key = semantic_scholar_api_key
        self.semantic_scholar_base_url = "https://api.semanticscholar.org/graph/v1"
        self.core_api_url = "https://api.core.ac.uk/v3"
        self.llm_service = llm_service
        
        # Semantic Scholar headers
        self.ss_headers = {}
        if semantic_scholar_api_key:
            self.ss_headers["x-api-key"] = semantic_scholar_api_key
            logger.info("✓ Semantic Scholar initialized with API key (primary)")
        else:
            logger.warning("⚠️ Semantic Scholar initialized WITHOUT API key (rate limits apply)")
        
        # arXiv client (no API key needed)
        self.arxiv_client = arxiv.Client()
        logger.info("✓ arXiv API initialized (fallback 1)")
        
        logger.info("✓ CORE API ready (fallback 2)")
    
    def search_papers(
        self,
        query: str,
        limit: int = 5,
        year_from: Optional[int] = None,
        min_citations: Optional[int] = None,
        use_query_rewriting: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for academic papers with automatic fallback.
        
        Args:
            query: Search query
            limit: Maximum number of papers
            year_from: Minimum year filter
            min_citations: Minimum citation count
            use_query_rewriting: Use LLM query rewriting
        
        Returns:
            List of paper dictionaries
        """
        # Query rewriting (keep existing logic)
        search_query = query
        if use_query_rewriting and self.llm_service:
            try:
                rewrite_result = self._rewrite_query(query)
                search_query = rewrite_result.get("primary_query", query)
                logger.info(f"Query rewritten: '{query}' → '{search_query}'")
            except Exception as e:
                logger.warning(f"Query rewriting failed: {e}, using original")
        
        # ATTEMPT 1: Semantic Scholar (best quality, rate-limited)
        if settings.enable_research_fallback:
            try:
                logger.info(f"🔍 ATTEMPT 1: Semantic Scholar search for: {search_query}")
                papers = self._search_semantic_scholar(search_query, limit, year_from)
                if papers:
                    logger.info(f"✅ Semantic Scholar returned {len(papers)} papers")
                    return self._format_papers(papers, "semantic_scholar")
                logger.warning("Semantic Scholar returned 0 results, trying arXiv...")
            except Exception as e:
                logger.warning(f"⚠️ Semantic Scholar failed: {e}, falling back to arXiv")
        
        # ATTEMPT 2: arXiv (free, unlimited, CS/Physics/Math focus)
        if settings.use_arxiv_fallback:
            try:
                logger.info(f"🔍 ATTEMPT 2: arXiv search for: {search_query}")
                papers = self._search_arxiv(search_query, limit, year_from)
                if papers:
                    logger.info(f"✅ arXiv returned {len(papers)} papers")
                    return papers
                logger.warning("arXiv returned 0 results, trying CORE...")
            except Exception as e:
                logger.warning(f"⚠️ arXiv search failed: {e}, falling back to CORE")
        
        # ATTEMPT 3: CORE (1,000/day free, open access focus)
        if settings.use_core_fallback:
            try:
                logger.info(f"🔍 ATTEMPT 3: CORE API search for: {search_query}")
                papers = self._search_core(search_query, limit, year_from)
                if papers:
                    logger.info(f"✅ CORE returned {len(papers)} papers")
                    return papers
                logger.warning("CORE returned 0 results")
            except Exception as e:
                logger.error(f"⚠️ CORE search failed: {e}")
        
        # All attempts exhausted
        logger.error("❌ All research search providers failed")
        return []
    
    def _search_semantic_scholar(
        self,
        query: str,
        limit: int,
        year_from: Optional[int]
    ) -> List[Dict]:
        """Search Semantic Scholar API."""
        try:
            url = f"{self.semantic_scholar_base_url}/paper/search"
            params = {
                "query": query,
                "limit": limit,
                "fields": "paperId,title,abstract,year,authors,citationCount,venue,url,openAccessPdf,tldr"
            }
            
            if year_from:
                params["year"] = f"{year_from}-"
            
            response = requests.get(url, params=params, headers=self.ss_headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [])
            elif response.status_code == 429:
                logger.error("Semantic Scholar rate limit exceeded")
                raise Exception("Rate limit exceeded")
            else:
                raise Exception(f"HTTP {response.status_code}")
        
        except Exception as e:
            logger.error(f"Semantic Scholar error: {e}")
            raise
    
    def _search_arxiv(
        self,
        query: str,
        limit: int,
        year_from: Optional[int]
    ) -> List[Dict[str, Any]]:
        """
        Search arXiv API (free, unlimited, 3 sec throttle).
        
        NOTE: arXiv is best for CS, Physics, Math papers.
        """
        try:
            # Build arXiv search
            search = arxiv.Search(
                query=query,
                max_results=limit,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            # Throttle to respect 3 second/call limit
            time.sleep(3)
            
            results = list(self.arxiv_client.results(search))
            
            # Filter by year if specified
            if year_from:
                results = [r for r in results if r.published.year >= year_from]
            
            # Format to standard schema
            papers = []
            for result in results[:limit]:
                papers.append({
                    "paperId": result.entry_id,
                    "title": result.title,
                    "abstract": result.summary,
                    "year": result.published.year,
                    "authors": [{"name": str(author)} for author in result.authors],
                    "citationCount": 0,  # arXiv doesn't provide citations
                    "venue": "arXiv",
                    "url": result.entry_id,
                    "openAccessPdf": {"url": result.pdf_url} if result.pdf_url else None,
                    "tldr": None,
                    "source": "arxiv"
                })
            
            return papers
        
        except Exception as e:
            logger.error(f"arXiv API error: {e}")
            raise
    
    def _search_core(
        self,
        query: str,
        limit: int,
        year_from: Optional[int]
    ) -> List[Dict[str, Any]]:
        """
        Search CORE API (1,000 requests/day free).
        
        NOTE: CORE focuses on open access papers across all disciplines.
        """
        try:
            url = f"{self.core_api_url}/search/works"
            params = {
                "q": query,
                "limit": limit,
                "scroll": False
            }
            
            # CORE API is free but requires no authentication for basic access
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                # Filter by year if specified
                if year_from:
                    results = [r for r in results if r.get("yearPublished", 0) >= year_from]
                
                # Format to standard schema
                papers = []
                for result in results[:limit]:
                    papers.append({
                        "paperId": result.get("id"),
                        "title": result.get("title", "Unknown"),
                        "abstract": result.get("abstract", "No abstract available"),
                        "year": result.get("yearPublished"),
                        "authors": [{"name": author} for author in result.get("authors", [])],
                        "citationCount": 0,  # CORE doesn't provide citation counts
                        "venue": result.get("publisher", "Unknown"),
                        "url": result.get("downloadUrl") or result.get("sourceFulltextUrls", [""])[0],
                        "openAccessPdf": {"url": result.get("downloadUrl")} if result.get("downloadUrl") else None,
                        "tldr": None,
                        "source": "core"
                    })
                
                return papers
            else:
                raise Exception(f"HTTP {response.status_code}")
        
        except Exception as e:
            logger.error(f"CORE API error: {e}")
            raise
    
    def _format_papers(self, papers: List[Dict], source: str) -> List[Dict]:
        """Add source metadata to papers."""
        for paper in papers:
            paper["source"] = source
        return papers
    
    def _rewrite_query(self, query: str) -> Dict[str, Any]:
        """Query rewriting logic (simplified from your existing code)."""
        # Extract keywords
        stopwords = ['compare', 'uploaded', 'paper', 'with', 'latest', 'findings', 'in', 'the', 'field', 'of']
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        primary_query = ' '.join(keywords[:8])
        
        return {
            "primary_query": primary_query,
            "sub_queries": [primary_query],
            "keywords": keywords[:5],
            "year_filter": None
        }
    
    

    def format_paper_for_context(self, paper: Dict[str, Any]) -> str:
        """
        Format paper for LLM context with None-safe handling.
        
        FIXED: Handle None values for abstract, tldr, and other fields.
        """
        title = paper.get("title", "Unknown Title")
        authors = paper.get("authors", [])
        
        # Safe author name extraction
        author_names = ", ".join([a.get("name", "") for a in authors[:3] if isinstance(a, dict)])
        if len(authors) > 3:
            author_names += " et al."
        if not author_names:
            author_names = "Unknown Authors"
        
        year = paper.get("year", "N/A")
        
        # FIXED: Safe abstract handling
        abstract = paper.get("abstract") or "No abstract available"
        if not isinstance(abstract, str):
            abstract = "No abstract available"
        
        citations = paper.get("citationCount", 0)
        venue = paper.get("venue", "Unknown Venue")
        source = paper.get("source", "unknown")
        
        # FIXED: Safe TLDR handling
        tldr = paper.get("tldr", {})
        tldr_text = ""
        if isinstance(tldr, dict):
            tldr_text = tldr.get("text", "")
        elif isinstance(tldr, str):
            tldr_text = tldr
        
        # Build context string
        context = f"""**Research Paper: {title}**
    Authors: {author_names}
    Year: {year} | Venue: {venue} | Citations: {citations} | Source: {source}
    {f"Summary: {tldr_text}" if tldr_text else ""}
    Abstract: {abstract[:500]}{"..." if len(abstract) > 500 else ""}
    """
        
        return context.strip()



# Global instance
_research_service = None

def get_research_search_service(api_key: Optional[str] = None, llm_service=None) -> MultiProviderResearchSearch:
    """Get or create global research search service instance."""
    global _research_service
    if _research_service is None:
        _research_service = MultiProviderResearchSearch(api_key, llm_service)
    return _research_service
