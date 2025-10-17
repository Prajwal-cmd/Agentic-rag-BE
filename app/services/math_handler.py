"""
Mathematical Formula Handler for Research Papers

Pattern: LaTeX parsing + KaTeX rendering + LLM verification
Source: Notion (KaTeX), Overleaf (LaTeX), OpenAI GPT-4 math (2024)

Features:
- Extract math from text/PDFs
- Convert to LaTeX
- Verify correctness with LLM
- Render-ready for frontend (KaTeX)
"""

from typing import List, Dict, Any, Optional, Tuple
import re
from pylatexenc.latex2text import LatexNodes2Text

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class MathFormulaHandler:
    """
    Handle mathematical formulas in research papers.
    
    Pattern: Extract → Convert → Verify → Render
    Source: KaTeX (rendering), pylatexenc (parsing)
    """
    
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self.latex_converter = LatexNodes2Text()
        logger.info("✓ Math Formula Handler initialized")
    
    def extract_math_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract mathematical expressions from text.
        
        Detects:
        - LaTeX: $...$ or $$...$$
        - Unicode math symbols
        - Common patterns (equations, formulas)
        
        Returns:
            [
                {
                    "original": "$E = mc^2$",
                    "latex": "E = mc^2",
                    "type": "inline" or "block",
                    "position": (start, end)
                }
            ]
        """
        formulas = []
        
        # Pattern 1: LaTeX inline ($...$)
        inline_pattern = r'\$([^\$]+)\$'
        for match in re.finditer(inline_pattern, text):
            formulas.append({
                "original": match.group(0),
                "latex": match.group(1).strip(),
                "type": "inline",
                "position": match.span(),
                "confidence": 1.0
            })
        
        # Pattern 2: LaTeX display ($$...$$)
        block_pattern = r'\$\$([^\$]+)\$\$'
        for match in re.finditer(block_pattern, text):
            formulas.append({
                "original": match.group(0),
                "latex": match.group(1).strip(),
                "type": "block",
                "position": match.span(),
                "confidence": 1.0
            })
        
        # Pattern 3: LaTeX environments (\begin{equation}...\end{equation})
        env_pattern = r'\\begin\{(equation|align|gather)\}(.*?)\\end\{\1\}'
        for match in re.finditer(env_pattern, text, re.DOTALL):
            formulas.append({
                "original": match.group(0),
                "latex": match.group(2).strip(),
                "type": "block",
                "position": match.span(),
                "environment": match.group(1),
                "confidence": 1.0
            })
        
        # Pattern 4: Unicode math (fallback detection)
        unicode_math_pattern = r'[∑∫∏√∞±×÷≤≥≠≈∈∉⊂⊃∪∩∧∨¬∀∃∂∇α-ωΑ-Ω]+'
        for match in re.finditer(unicode_math_pattern, text):
            # Only add if not already in formulas
            if not any(f["position"][0] <= match.start() < f["position"][1] for f in formulas):
                formulas.append({
                    "original": match.group(0),
                    "latex": self._unicode_to_latex(match.group(0)),
                    "type": "inline",
                    "position": match.span(),
                    "confidence": 0.7  # Lower confidence for auto-conversion
                })
        
        logger.info(f"Extracted {len(formulas)} mathematical expressions")
        return formulas
    
    def _unicode_to_latex(self, unicode_text: str) -> str:
        """Convert common Unicode math symbols to LaTeX."""
        conversions = {
            '∑': r'\sum',
            '∫': r'\int',
            '∏': r'\prod',
            '√': r'\sqrt',
            '∞': r'\infty',
            '±': r'\pm',
            '×': r'\times',
            '÷': r'\div',
            '≤': r'\leq',
            '≥': r'\geq',
            '≠': r'\neq',
            '≈': r'\approx',
            '∈': r'\in',
            '∉': r'\notin',
            '⊂': r'\subset',
            '⊃': r'\supset',
            '∪': r'\cup',
            '∩': r'\cap',
            '∧': r'\land',
            '∨': r'\lor',
            '¬': r'\neg',
            '∀': r'\forall',
            '∃': r'\exists',
            '∂': r'\partial',
            '∇': r'\nabla',
            # Greek letters
            'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta',
            'ε': r'\epsilon', 'ζ': r'\zeta', 'η': r'\eta', 'θ': r'\theta',
            'λ': r'\lambda', 'μ': r'\mu', 'π': r'\pi', 'σ': r'\sigma',
            'τ': r'\tau', 'φ': r'\phi', 'ω': r'\omega'
        }
        
        result = unicode_text
        for unicode_char, latex in conversions.items():
            result = result.replace(unicode_char, latex)
        
        return result
    
    def verify_latex_syntax(self, latex: str) -> Tuple[bool, Optional[str]]:
        """
        Verify LaTeX syntax is valid.
        
        Returns:
            (is_valid, error_message)
        """
        try:
            # Try to parse with pylatexenc
            self.latex_converter.latex_to_text(latex)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def fix_latex_with_llm(self, broken_latex: str) -> Optional[str]:
        """
        Use LLM to fix broken LaTeX expressions.
        
        Pattern: GPT-4 math correction (OpenAI 2024)
        """
        if not self.llm_service:
            logger.warning("LLM service not available for LaTeX correction")
            return None
        
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a LaTeX math expert. Fix the broken LaTeX expression below.

Rules:
1. Only return the corrected LaTeX, nothing else
2. Do not add $$ delimiters
3. Fix syntax errors, missing braces, etc.
4. Keep the mathematical meaning intact

If the expression is too broken to fix, return: ERROR"""),
            ("human", "Broken LaTeX: {latex}\n\nFixed LaTeX:")
        ])
        
        llm = self.llm_service.get_llm("llama-3.1-8b-instant", temperature=0.1)
        chain = prompt | llm | StrOutputParser()
        
        try:
            fixed = chain.invoke({"latex": broken_latex})
            
            if fixed.strip() == "ERROR":
                return None
            
            return fixed.strip()
        
        except Exception as e:
            logger.error(f"LLM LaTeX correction failed: {e}")
            return None
    
    def normalize_for_rendering(self, latex: str, mode: str = "inline") -> str:
        """
        Normalize LaTeX for KaTeX rendering (frontend).
        
        Args:
            latex: LaTeX string
            mode: "inline" or "block"
        
        Returns:
            Render-ready LaTeX string
        """
        # Remove common problematic commands for KaTeX
        latex = latex.strip()
        
        # Remove \label{} (not needed for rendering)
        latex = re.sub(r'\\label\{[^}]*\}', '', latex)
        
        # Replace \[ \] with $$ $$
        latex = latex.replace(r'\[', '').replace(r'\]', '')
        
        # Normalize whitespace
        latex = re.sub(r'\s+', ' ', latex).strip()
        
        return latex
    
    def format_for_llm_context(self, formulas: List[Dict[str, Any]]) -> str:
        """
        Format extracted formulas for LLM context.
        
        Returns formatted string with all formulas.
        """
        if not formulas:
            return "No mathematical formulas found."
        
        formatted = "**Mathematical Formulas:**\n\n"
        
        for i, formula in enumerate(formulas, 1):
            latex = formula["latex"]
            formula_type = formula["type"]
            
            if formula_type == "inline":
                formatted += f"{i}. $${latex}$$\n"
            else:
                formatted += f"{i}. Block formula:\n   $${latex}$$\n"
        
        return formatted

# Global instance
_math_handler = None

def get_math_handler(llm_service=None) -> MathFormulaHandler:
    """Get or create math handler singleton."""
    global _math_handler
    if _math_handler is None:
        _math_handler = MathFormulaHandler(llm_service)
    return _math_handler
