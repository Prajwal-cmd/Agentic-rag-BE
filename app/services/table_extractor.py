"""
Balanced Table Extraction
Filters false positives while detecting real tables
"""
from typing import List, Dict, Any, Optional
import io
import base64
import re

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

import pandas as pd
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class TableExtractor:
    """
    Balanced table extraction with smart filtering.
    """
    
    def __init__(self):
        self.supported_methods = []
        if CAMELOT_AVAILABLE:
            self.supported_methods.append("camelot")
            logger.info("✓ Camelot available")
        if PDFPLUMBER_AVAILABLE:
            self.supported_methods.append("pdfplumber")
            logger.info("✓ pdfplumber available")
    
    def extract_tables_from_pdf(
        self,
        pdf_path: str,
        pages: str = "all"
    ) -> List[Dict[str, Any]]:
        """
        Extract tables with balanced filtering.
        Allows 2+ columns OR special single-column tables.
        """
        tables = []
        
        # Try pdfplumber first
        if "pdfplumber" in self.supported_methods:
            logger.info(f"🔍 Extracting from {pdf_path}")
            try:
                tables = self._extract_with_pdfplumber(pdf_path, pages)
                if tables:
                    logger.info(f"✅ Found {len(tables)} tables")
                    return tables
            except Exception as e:
                logger.error(f"pdfplumber failed: {e}")
        
        # Fallback to Camelot
        if "camelot" in self.supported_methods:
            logger.info("🔍 Trying Camelot fallback")
            try:
                tables = self._extract_with_camelot(pdf_path, pages)
                if tables:
                    logger.info(f"✅ Camelot found {len(tables)} tables")
                    return tables
            except Exception as e:
                logger.error(f"Camelot failed: {e}")
        
        logger.warning("⚠️ No valid tables found")
        return []
    
    def _extract_with_pdfplumber(
        self,
        pdf_path: str,
        pages: str
    ) -> List[Dict[str, Any]]:
        """Extract with pdfplumber."""
        validated_tables = []
        table_number = 1
        
        with pdfplumber.open(pdf_path) as pdf:
            if pages == "all":
                page_numbers = range(len(pdf.pages))
            else:
                page_numbers = self._parse_pages(pages, len(pdf.pages))
            
            for page_num in page_numbers:
                try:
                    page = pdf.pages[page_num]
                    
                    # Use line-based detection
                    table_settings = {
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "snap_tolerance": 3,
                        "join_tolerance": 3,
                    }
                    
                    tables = page.find_tables(table_settings=table_settings)
                    
                    for table_obj in tables:
                        try:
                            table_data = table_obj.extract()
                            
                            if not table_data or len(table_data) < 2:
                                continue
                            
                            # Convert to DataFrame
                            headers = table_data[0]
                            data_rows = table_data[1:]
                            
                            headers = [str(h).strip() if h else f"Col_{i}" for i, h in enumerate(headers)]
                            df = pd.DataFrame(data_rows, columns=headers)
                            
                            # Clean
                            df = df.replace('', pd.NA).replace(None, pd.NA)
                            df = df.dropna(how='all').dropna(axis=1, how='all')
                            df = df.reset_index(drop=True)
                            
                            # BALANCED VALIDATION - less strict
                            if self._is_valid_table(df):
                                validated_tables.append({
                                    "data": df,
                                    "page": page_num + 1,
                                    "table_number": table_number,
                                    "method": "pdfplumber"
                                })
                                table_number += 1
                            else:
                                logger.debug(f"Rejected on page {page_num + 1}: {len(df)} rows, {len(df.columns)} cols")
                            
                        except Exception as e:
                            logger.debug(f"Table processing error: {e}")
                            continue
                            
                except Exception as e:
                    logger.warning(f"Page {page_num + 1} error: {e}")
                    continue
        
        return validated_tables
    
    def _extract_with_camelot(
        self,
        pdf_path: str,
        pages: str
    ) -> List[Dict[str, Any]]:
        """Extract with Camelot."""
        validated_tables = []
        
        try:
            # Try lattice
            tables_raw = camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor="lattice",
                strip_text='\n',
                line_scale=40,
                suppress_stdout=True
            )
            
            for idx, table in enumerate(tables_raw):
                df = table.df
                
                if df.empty:
                    continue
                
                # Clean
                df = df.replace('', pd.NA).dropna(how='all').dropna(axis=1, how='all')
                df = df.reset_index(drop=True)
                
                if df.empty:
                    continue
                
                # Balanced validation
                if self._is_valid_table(df):
                    # Use first row as headers if applicable
                    if len(df) > 1 and self._looks_like_header(df.iloc[0]):
                        df.columns = df.iloc[0].astype(str)
                        df = df[1:].reset_index(drop=True)
                    
                    validated_tables.append({
                        "data": df,
                        "page": table.page,
                        "table_number": idx + 1,
                        "method": "camelot"
                    })
                else:
                    logger.debug(f"Camelot rejected table {idx + 1}: {len(df)} rows, {len(df.columns)} cols")
            
            return validated_tables
            
        except Exception as e:
            logger.error(f"Camelot error: {e}")
            return []
    
    def _is_valid_table(self, df: pd.DataFrame) -> bool:
        """
        BALANCED validation - filters obvious non-tables only.
        
        Rejects:
        - Less than 2 rows
        - Single column with paragraph text
        - All empty
        
        Accepts:
        - 2+ columns with any data
        - Single column with structured data (lists, short items)
        """
        # Minimum 2 rows
        if len(df) < 2:
            return False
        
        # Must have at least some data
        total_cells = len(df) * len(df.columns)
        filled_cells = df.notna().sum().sum()
        
        if filled_cells < total_cells * 0.1:  # At least 10% filled
            return False
        
        # Multi-column tables are usually valid
        if len(df.columns) >= 2:
            return True
        
        # Single column - check if it's paragraph text or structured data
        if len(df.columns) == 1:
            # Calculate average cell length
            cell_lengths = [len(str(v)) for v in df.iloc[:, 0] if pd.notna(v)]
            
            if not cell_lengths:
                return False
            
            avg_length = sum(cell_lengths) / len(cell_lengths)
            
            # If average cell > 200 chars, it's probably paragraph text
            if avg_length > 200:
                logger.debug(f"Rejected single-column: avg length {avg_length:.0f} (paragraph text)")
                return False
            
            # Check if it looks like a list (structured data)
            # Lists have: short items, some repetition pattern, numbers/bullets
            sample_text = ' '.join([str(v) for v in df.iloc[:, 0].head(5) if pd.notna(v)])
            
            # Check for list-like patterns
            has_numbers = bool(re.search(r'^\d+\.?\s', sample_text, re.MULTILINE))
            has_bullets = any(marker in sample_text for marker in ['•', '◦', '▪', '-', '*'])
            
            # If it has numbering or bullets, it might be a structured list table
            if has_numbers or has_bullets:
                return True
            
            # Otherwise, if avg cell is > 100 chars, probably not a table
            if avg_length > 100:
                logger.debug(f"Rejected single-column: avg length {avg_length:.0f} (long text)")
                return False
        
        return True
    
    def _looks_like_header(self, row: pd.Series) -> bool:
        """Check if row looks like headers."""
        try:
            non_numeric = sum(
                1 for v in row 
                if pd.notna(v) and not str(v).replace('.', '').replace('-', '').isdigit()
            )
            return non_numeric > len(row) / 2
        except:
            return False
    
    def _parse_pages(self, pages_str: str, total_pages: int) -> List[int]:
        """Parse page string."""
        page_numbers = []
        for part in pages_str.split(','):
            try:
                part = part.strip()
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    page_numbers.extend(range(start - 1, min(end, total_pages)))
                else:
                    page_num = int(part) - 1
                    if 0 <= page_num < total_pages:
                        page_numbers.append(page_num)
            except:
                continue
        return sorted(set(page_numbers))
    
    def export_tables(
        self,
        tables: List[Dict[str, Any]],
        output_format: str = "csv"
    ) -> List[Dict[str, Any]]:
        """Export tables."""
        exported = []
        
        for table in tables:
            try:
                df = table["data"]
                ext = "csv" if output_format == "csv" else "xlsx"
                filename = f"table_p{table['page']}_n{table['table_number']}.{ext}"
                
                if output_format == "csv":
                    buffer = io.BytesIO()
                    df.to_csv(buffer, index=False, encoding='utf-8-sig')
                    file_content = buffer.getvalue()
                else:
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name=f'Table_{table["table_number"]}')
                    file_content = buffer.getvalue()
                
                exported.append({
                    "filename": filename,
                    "content": base64.b64encode(file_content).decode('utf-8'),
                    "page": table["page"],
                    "table_number": table["table_number"],
                    "rows": len(df),
                    "columns": len(df.columns),
                    "format": output_format
                })
                
                logger.info(f"✅ Exported {filename} ({len(df)}×{len(df.columns)})")
            except Exception as e:
                logger.error(f"Export error: {e}")
                continue
        
        return exported
    
    def tables_to_markdown(self, tables: List[Dict[str, Any]]) -> str:
        """Convert to markdown."""
        parts = []
        for table in tables:
            try:
                df = table["data"]
                parts.append(f"\n### Table {table['table_number']} (Page {table['page']})\n")
                parts.append(df.to_markdown(index=False))
                parts.append("\n")
            except:
                continue
        return "\n".join(parts)
    
    def get_table_summary(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary."""
        if not tables:
            return {"total_tables": 0}
        
        return {
            "total_tables": len(tables),
            "pages_with_tables": len(set(t["page"] for t in tables)),
            "methods_used": list(set(t.get("method", "unknown") for t in tables)),
            "total_rows": sum(len(t["data"]) for t in tables),
            "total_columns": sum(len(t["data"].columns) for t in tables),
            "tables_by_page": {str(t["page"]): 1 for t in tables}
        }


_table_extractor = None

def get_table_extractor():
    global _table_extractor
    if _table_extractor is None:
        _table_extractor = TableExtractor()
    return _table_extractor
