"""
Literature Review Service - Elicit-Style Research Automation

Pattern: Multi-paper synthesis with structured extraction
Source: Elicit (2025), Semantic Scholar Research (2024)

Features:
- Automated paper discovery & ranking
- Multi-paper synthesis & comparison
- Citation export (BibTeX/RIS)
- Key findings extraction
"""

from typing import List, Dict, Any, Optional
import json
from dataclasses import dataclass
from datetime import datetime

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class ResearchPaper:
    """Structured paper representation for literature review."""
    paper_id: str
    title: str
    authors: List[str]
    year: int
    abstract: str
    citations: int
    venue: str
    url: str
    key_findings: List[str] = None
    methodology: str = None
    limitations: str = None
    source: str = "semantic_scholar"

class LiteratureReviewService:
    """
    Elicit-style literature review automation.
    
    Pattern: Structured extraction + synthesis (not just summarization)
    Source: Elicit whitepaper (2024), Ought research
    """
    
    def __init__(self, llm_service, research_service):
        self.llm_service = llm_service
        self.research_service = research_service
        logger.info("✓ Literature Review Service initialized")
    
    def conduct_literature_review(
        self,
        research_question: str,
        max_papers: int = 10,
        min_year: int = 2020,
        extract_structured_data: bool = True
    ) -> Dict[str, Any]:
        """
        Conduct automated literature review on a research question.
        
        Pattern: Elicit's 4-step process:
        1. Query decomposition
        2. Paper retrieval & ranking
        3. Structured extraction (key findings, methods, limitations)
        4. Synthesis across papers
        
        Args:
            research_question: Research question to investigate
            max_papers: Maximum papers to analyze
            min_year: Minimum publication year
            extract_structured_data: Extract structured info from each paper
        
        Returns:
            {
                "research_question": str,
                "papers": List[ResearchPaper],
                "synthesis": str,
                "key_themes": List[str],
                "research_gaps": List[str],
                "citation_export": Dict[str, str]  # BibTeX, RIS formats
            }
        """
        logger.info(f"Starting literature review for: {research_question}")
        
        try:
            # STEP 1: Retrieve papers using existing research service
            papers = self.research_service.search_papers(
                query=research_question,
                limit=max_papers,
                year_from=min_year,
                use_query_rewriting=True
            )
            
            if not papers:
                return self._empty_review_result(
                    research_question,
                    error_type="no_results",
                    error_message="No papers found matching your query. Try rephrasing or broadening the search terms."
                )
            
            # STEP 2: Convert to structured format
            structured_papers = [self._convert_to_research_paper(p) for p in papers]
            
            # STEP 3: Extract structured data from each paper (if enabled)
            if extract_structured_data:
                for paper in structured_papers:
                    extracted_data = self._extract_structured_info(paper)
                    paper.key_findings = extracted_data.get("key_findings", [])
                    paper.methodology = extracted_data.get("methodology", "")
                    paper.limitations = extracted_data.get("limitations", "")
            
            # STEP 4: Synthesize findings across papers
            synthesis = self._synthesize_findings(research_question, structured_papers)
            
            # STEP 5: Identify key themes and gaps
            themes = self._extract_themes(structured_papers, synthesis)
            gaps = self._identify_research_gaps(structured_papers, synthesis)
            
            # STEP 6: Generate citation exports
            citations = self._generate_citation_exports(structured_papers)
            
            logger.info(f"✅ Literature review complete: {len(structured_papers)} papers analyzed")
            
            return {
                "research_question": research_question,
                "papers": [self._paper_to_dict(p) for p in structured_papers],
                "synthesis": synthesis,
                "key_themes": themes,
                "research_gaps": gaps,
                "citation_export": citations,
                "metadata": {
                    "total_papers": len(structured_papers),
                    "date_generated": datetime.now().isoformat(),
                    "year_range": f"{min_year}-{datetime.now().year}",
                    "status": "success"
                }
            }
        
        except Exception as e:
            error_message = str(e)
            error_type = "unknown_error"
            
            # Detect specific error types for user-friendly messages
            if "rate limit" in error_message.lower() or "429" in error_message:
                error_type = "rate_limit"
                error_message = "API rate limit exceeded. Please try again in a few minutes. The system will automatically use arXiv and CORE as fallback sources."
            
            elif "timeout" in error_message.lower():
                error_type = "timeout"
                error_message = "Request timed out. The academic databases are temporarily slow. Please try again in a moment."
            
            elif "connection" in error_message.lower() or "network" in error_message.lower():
                error_type = "connection_error"
                error_message = "Unable to connect to research databases. Please check your internet connection and try again."
            
            elif "not found" in error_message.lower():
                error_type = "no_results"
                error_message = f"No papers found for: {research_question}. Try rephrasing or broadening your search terms."
            
            logger.error(f"Literature review failed [{error_type}]: {error_message}")
            return self._empty_review_result(
                research_question,
                error_type=error_type,
                error_message=error_message
            )



    def _convert_to_research_paper(self, raw_paper: Dict) -> ResearchPaper:
        """Convert raw API response to structured ResearchPaper."""
        authors = [a.get("name", "Unknown") for a in raw_paper.get("authors", [])]
        
        return ResearchPaper(
            paper_id=raw_paper.get("paperId", ""),
            title=raw_paper.get("title", "Unknown"),
            authors=authors[:5],  # Limit to 5 authors
            year=raw_paper.get("year", 0),
            abstract=raw_paper.get("abstract", ""),
            citations=raw_paper.get("citationCount", 0),
            venue=raw_paper.get("venue", "Unknown"),
            url=raw_paper.get("url", ""),
            source=raw_paper.get("source", "semantic_scholar")
        )
    
    def _extract_structured_info(self, paper: ResearchPaper) -> Dict[str, Any]:
        """
        Extract structured information from paper abstract using LLM.
        
        Pattern: Elicit's structured extraction (not summarization)
        Extracts: Key findings, methodology, limitations
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert research analyst. Extract structured information from this research paper abstract.

Return a JSON object with exactly three fields:
- key_findings: an array of 2-4 key findings as strings
- methodology: a brief string describing the research method used
- limitations: a string describing key limitations or gaps mentioned

Be concise and factual. If information is not in the abstract, use empty string or empty array."""),
                ("human", """Paper: {title}
Authors: {authors}
Year: {year}

Abstract: {abstract}

Extract structured information in JSON format.""")
            ])
            
            llm = self.llm_service.get_llm("llama-3.1-8b-instant", temperature=0.1)
            parser = JsonOutputParser()
            chain = prompt | llm | parser
            
            result = chain.invoke({
                "title": paper.title,
                "authors": ", ".join(paper.authors[:3]),
                "year": paper.year,
                "abstract": paper.abstract[:800]  # Limit context
            })
            
            return result
        
        except Exception as e:
            logger.warning(f"Structured extraction failed for {paper.title}: {e}")
            return {
                "key_findings": [],
                "methodology": "",
                "limitations": ""
            }
    
    def _synthesize_findings(
        self,
        research_question: str,
        papers: List[ResearchPaper]
    ) -> str:
        """
        Synthesize findings across multiple papers.
        
        Pattern: Cross-paper synthesis (Elicit approach)
        NOT just concatenating abstracts
        """
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Build paper summaries
        paper_summaries = []
        for i, paper in enumerate(papers[:10], 1):  # Limit to 10 for context
            summary = f"""{i}. **{paper.title}** ({paper.year})
   Authors: {', '.join(paper.authors[:3])} {'et al.' if len(paper.authors) > 3 else ''}
   Citations: {paper.citations} | Venue: {paper.venue}
   
   Key Findings: {', '.join(paper.key_findings) if paper.key_findings else 'Not extracted'}
   Methodology: {paper.methodology or 'Not specified'}
"""
            paper_summaries.append(summary)
        
        context = "\n\n".join(paper_summaries)
        
        prompt = ChatPromptTemplate.from_template("""You are an expert research synthesizer. Analyze these research papers to answer the research question.

**Research Question:** {research_question}

**Papers:**
{context}

**Instructions:**
1. Synthesize findings ACROSS papers (not paper-by-paper summary)
2. Identify consensus and contradictions
3. Note methodological approaches
4. Highlight high-impact findings (high citations)
5. Structure as: Overview → Main Findings → Methodologies → Conclusions

Provide a comprehensive synthesis (400-600 words):""")
        
        llm = self.llm_service.get_llm("llama-3.3-70b-versatile", temperature=0.3)
        chain = prompt | llm | StrOutputParser()
        
        synthesis = chain.invoke({
            "research_question": research_question,
            "context": context
        })
        
        return synthesis
    
    def _extract_themes(
        self,
        papers: List[ResearchPaper],
        synthesis: str
    ) -> List[str]:
        """Extract key themes from literature."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract 3-5 key research themes from this literature synthesis. Return JSON array of strings."),
            ("human", "Synthesis:\n{synthesis}\n\nExtract key themes (JSON array):")
        ])
        
        llm = self.llm_service.get_llm("llama-3.1-8b-instant", temperature=0.2)
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        try:
            themes = chain.invoke({"synthesis": synthesis[:2000]})
            return themes if isinstance(themes, list) else []
        except:
            return []
    
    def _identify_research_gaps(
        self,
        papers: List[ResearchPaper],
        synthesis: str
    ) -> List[str]:
        """Identify research gaps from literature."""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        
        limitations = [p.limitations for p in papers if p.limitations]
        
        if not limitations:
            return []
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Identify 2-4 research gaps or limitations from this literature. Return JSON array of strings."),
            ("human", "Limitations mentioned:\n{limitations}\n\nResearch gaps (JSON array):")
        ])
        
        llm = self.llm_service.get_llm("llama-3.1-8b-instant", temperature=0.2)
        parser = JsonOutputParser()
        chain = prompt | llm | parser
        
        try:
            gaps = chain.invoke({"limitations": "\n".join(limitations[:10])})
            return gaps if isinstance(gaps, list) else []
        except:
            return []
    
    def _generate_citation_exports(
        self,
        papers: List[ResearchPaper]
    ) -> Dict[str, str]:
        """
        Generate citation exports in multiple formats.
        
        Formats: BibTeX, RIS (compatible with Zotero, Mendeley, EndNote)
        """
        bibtex_entries = []
        ris_entries = []
        
        for i, paper in enumerate(papers, 1):
            # BibTeX format
            # Clean title for BibTeX key
            key = f"{paper.authors[0].split()[-1].lower() if paper.authors else 'unknown'}{paper.year}"
            
            # Build author string separately to avoid f-string escaping issues
            authors_str = ' and '.join(paper.authors[:5])
            
            bibtex = f"""@article{{{key},
    title={{{paper.title}}},
    author={{{authors_str}}},
    year={{{paper.year}}},
    journal={{{paper.venue}}},
    url={{{paper.url}}},
    note={{Citations: {paper.citations}}}
}}"""
            bibtex_entries.append(bibtex)
            
            # RIS format (RefMan)
            author_lines = '\nAU  - '.join(paper.authors[:5])
            ris = f"""TY  - JOUR
TI  - {paper.title}
AU  - {author_lines}
PY  - {paper.year}
JO  - {paper.venue}
UR  - {paper.url}
N1  - Citations: {paper.citations}
ER  - """
            ris_entries.append(ris)
        
        return {
            "bibtex": "\n\n".join(bibtex_entries),
            "ris": "\n\n".join(ris_entries),
            "count": len(papers)
        }

    def _paper_to_dict(self, paper: ResearchPaper) -> Dict:
        """Convert ResearchPaper to dictionary."""
        return {
            "paper_id": paper.paper_id,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "abstract": paper.abstract,
            "citations": paper.citations,
            "venue": paper.venue,
            "url": paper.url,
            "key_findings": paper.key_findings or [],
            "methodology": paper.methodology or "",
            "limitations": paper.limitations or "",
            "source": paper.source
        }
    
    def _empty_review_result(
        self, 
        research_question: str, 
        error_type: str = None,
        error_message: str = None
    ) -> Dict:
        """
        Return empty result structure with enhanced error information.
        
        Args:
            research_question: The original research question
            error_type: Type of error (rate_limit, no_results, timeout, etc.)
            error_message: User-friendly error message
        
        Returns:
            Dict with empty results and error metadata
        """
        base_result = {
            "research_question": research_question,
            "papers": [],
            "synthesis": f"No papers found for: {research_question}",
            "key_themes": [],
            "research_gaps": [],
            "citation_export": {"bibtex": "", "ris": "", "count": 0},
            "metadata": {
                "total_papers": 0,
                "date_generated": datetime.now().isoformat(),
                "status": "error" if error_type else "no_results"
            }
        }
        
        # Add error details if present
        if error_type:
            base_result["metadata"]["error_type"] = error_type
            base_result["metadata"]["error_message"] = error_message
            
            # Add helpful suggestions based on error type
            if error_type == "rate_limit":
                base_result["metadata"]["suggestion"] = "Wait 5-10 minutes before retrying. The system uses arXiv and CORE as fallback sources."
                base_result["metadata"]["fallback_available"] = True
                base_result["metadata"]["retry_after"] = "5 minutes"
            
            elif error_type == "no_results":
                base_result["metadata"]["suggestion"] = "Try rephrasing your query, using different keywords, or broadening the search terms."
            
            elif error_type == "timeout":
                base_result["metadata"]["suggestion"] = "The databases are slow right now. Try again in a moment."
            
            elif error_type == "connection_error":
                base_result["metadata"]["suggestion"] = "Check your internet connection and ensure the backend server is running."
            
            else:
                base_result["metadata"]["suggestion"] = "Please try again later or contact support if the issue persists."
        
        return base_result

# Global instance
_lit_review_service = None

def get_literature_review_service(llm_service, research_service):
    """Get or create literature review service."""
    global _lit_review_service
    if _lit_review_service is None:
        _lit_review_service = LiteratureReviewService(llm_service, research_service)
    return _lit_review_service
