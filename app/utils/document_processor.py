"""
Document Processing Utilities

Pattern: Recursive chunking with contextual embedding support
Source: LangChain text splitters + Anthropic Contextual Retrieval
"""

from typing import List, Dict
import PyPDF2
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import BytesIO
from ..utils.logger import setup_logger
from ..config import settings

logger = setup_logger(__name__)

class DocumentProcessor:
    """
    Process various document formats into chunks for embedding.
    Now supports contextual embedding for improved retrieval.
    """
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Target chunk size in characters (default from config)
            chunk_overlap: Overlap between chunks for context preservation (default from config)
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # RecursiveCharacterTextSplitter - industry standard
        # Source: LangChain documentation
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " ", ""],  # Hierarchical splitting
            length_function=len
        )
        
        logger.info(f"DocumentProcessor initialized (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
    
    def extract_text_from_pdf(self, file_bytes: BytesIO) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file_bytes)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise ValueError(f"Failed to process PDF: {str(e)}")
    
    def extract_text_from_docx(self, file_bytes: BytesIO) -> str:
        """Extract text from DOCX file"""
        try:
            doc = DocxDocument(file_bytes)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise ValueError(f"Failed to process DOCX: {str(e)}")
    
    def extract_text_from_txt(self, file_bytes: BytesIO) -> str:
        """Extract text from TXT file"""
        try:
            return file_bytes.read().decode('utf-8')
        except Exception as e:
            logger.error(f"TXT extraction failed: {e}")
            raise ValueError(f"Failed to process TXT: {str(e)}")
    
    def process_document(self, file_bytes: BytesIO, filename: str, session_id: str) -> Dict:
        """
        Process document and return chunks with metadata + full text.
        
        ENHANCED: Returns full document text for contextual embedding.
        
        Args:
            file_bytes: Document file bytes (BytesIO object)
            filename: Original filename with extension
            session_id: User session ID for tracking
            
        Returns:
            Dict containing:
                - chunks: List of chunk dicts with text and metadata
                - full_text: Complete document text (for contextual embedding)
                - metadata: Document-level metadata
                
        Raises:
            ValueError: If file type unsupported or document empty
        """
        # Determine file type from extension
        file_extension = filename.lower().split('.')[-1] if '.' in filename else 'unknown'
        
        # Extract text based on file type
        try:
            if file_extension == 'pdf':
                text = self.extract_text_from_pdf(file_bytes)
            elif file_extension in ['docx', 'doc']:
                text = self.extract_text_from_docx(file_bytes)
            elif file_extension == 'txt':
                text = self.extract_text_from_txt(file_bytes)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}. Supported: PDF, DOCX, TXT")
        except Exception as e:
            logger.error(f"Text extraction failed for {filename}: {e}")
            raise ValueError(f"Failed to extract text from {filename}: {str(e)}")
        
        # Validate extracted text
        if not text or len(text.strip()) < 10:
            raise ValueError(f"Document appears to be empty or unreadable: {filename}")
        
        logger.info(f"Extracted {len(text)} characters from {filename}")
        
        # Split into chunks
        chunks_text = self.text_splitter.split_text(text)
        
        if not chunks_text:
            raise ValueError(f"Document chunking failed for {filename}")
        
        # Create chunk dictionaries with metadata
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            # Calculate relative position in document
            relative_position = i / max(len(chunks_text), 1)
            
            chunk = {
                "text": chunk_text,
                "metadata": {
                    "filename": filename,
                    "session_id": session_id,
                    "chunk_index": i,
                    "total_chunks": len(chunks_text),
                    "file_type": file_extension,
                    "relative_position": relative_position,  # NEW: 0.0 to 1.0
                    "is_first_chunk": (i == 0),  # NEW: Boolean flag
                    "is_last_chunk": (i == len(chunks_text) - 1),  # NEW: Boolean flag
                    "has_context": False,  # Will be updated if contextual embedding applied
                    "source": f"{filename} (chunk {i+1}/{len(chunks_text)})"  # NEW: Human-readable source
                }
            }
            chunks.append(chunk)
        
        logger.info(f"✅ Split document into {len(chunks)} chunks")
        
        # Document-level metadata
        doc_metadata = {
            "filename": filename,
            "session_id": session_id,
            "total_chunks": len(chunks),
            "file_type": file_extension,
            "char_count": len(text),
            "word_count": len(text.split()),  # NEW: Approximate word count
            "avg_chunk_size": len(text) // max(len(chunks), 1)  # NEW: Average chunk size
        }
        
        return {
            "chunks": chunks,
            "full_text": text,  # CRITICAL: Full text for contextual embedding
            "metadata": doc_metadata
        }




    def get_chunk_preview(self, chunk_text: str, max_length: int = 100) -> str:
        """Get a preview of chunk text for logging."""
        if len(chunk_text) <= max_length:
            return chunk_text
        return chunk_text[:max_length] + "..."


# Global instance with caching
from functools import lru_cache

@lru_cache(maxsize=1)
def get_document_processor() -> DocumentProcessor:
    """Get or create global document processor instance."""
    return DocumentProcessor()
