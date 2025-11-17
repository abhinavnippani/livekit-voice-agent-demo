"""
PDF Loading and Chunking
Handles PDF file loading, parsing, and text chunking
"""
import logging
from pathlib import Path
from typing import List
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

logger = logging.getLogger(__name__)


class PDFLoader:
    """
    Handles PDF loading and chunking operations.
    """
    
    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 30,
    ):
        """
        Initialize PDF loader.
        
        Args:
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Configure text splitter
        Settings.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a PDF file and return documents.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        logger.info("="*70)
        logger.info("PDF LOADER: LOADING PDF")
        logger.info("="*70)
        logger.info(f"   PDF path: {pdf_path}")
        
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            # Use PDFReader to load PDF
            reader = PDFReader()
            documents = reader.load_data(file=pdf_path_obj)
            logger.info(f"   ✅ Loaded {len(documents)} document(s) from PDF")
            
            # Log document info
            total_chars = sum(len(doc.text) for doc in documents)
            logger.info(f"   Total characters: {total_chars:,}")
            logger.info("="*70)
            
            return documents
        except Exception as e:
            logger.error(f"   ❌ ERROR loading PDF: {e}", exc_info=True)
            raise
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents into smaller pieces.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        logger.info("="*70)
        logger.info("PDF LOADER: CHUNKING DOCUMENTS")
        logger.info("="*70)
        logger.info(f"   Input documents: {len(documents)}")
        
        try:
            # Use the configured node parser to chunk documents
            node_parser = Settings.node_parser
            
            all_chunks = []
            for doc in documents:
                # Parse document into nodes (chunks)
                nodes = node_parser.get_nodes_from_documents([doc])
                all_chunks.extend(nodes)
            
            logger.info(f"   ✅ Created {len(all_chunks)} chunks")
            avg_size = sum(len(n.text) for n in all_chunks) / len(all_chunks) if all_chunks else 0
            logger.info(f"   Average chunk size: {avg_size:.0f} chars")
            logger.info("="*70)
            
            # Convert nodes back to documents for compatibility
            chunked_docs = [Document(text=node.text, metadata=node.metadata) for node in all_chunks]
            return chunked_docs
        except Exception as e:
            logger.error(f"   ❌ ERROR chunking documents: {e}", exc_info=True)
            raise
    
    def load_and_chunk_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a PDF and chunk it in one step.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of chunked Document objects
        """
        documents = self.load_pdf(pdf_path)
        chunked_docs = self.chunk_documents(documents)
        return chunked_docs

