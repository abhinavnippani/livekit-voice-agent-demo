"""
RAG Service - Main Orchestrator
Coordinates PDF loading, chunking, vector storage, and retrieval
"""
import logging
import os
from typing import Optional, List
from llama_index.core import Document, Settings

from .vector_store import VectorStoreManager
from .pdf_loader import PDFLoader
from .retrieval import Retriever

logger = logging.getLogger(__name__)


class RAGService:
    """
    Main RAG service that orchestrates all components.
    Uses FAISS for high-performance semantic search.
    """
    
    def __init__(
        self,
        vector_db_path: Optional[str] = None,
        collection_name: str = "voice_assistant_rag",
        chunk_size: int = 200,
        chunk_overlap: int = 20,
        top_k: int = 3,
    ):
        """
        Initialize RAG service with FAISS vector store.
        
        Args:
            vector_db_path: Path to store vector database (default: ./vector_db)
            collection_name: Name of the collection/namespace
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
            top_k: Number of top results to retrieve
        """
        logger.info("="*70)
        logger.info("RAG SERVICE: INITIALIZATION")
        logger.info("="*70)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Initialize embedding model
        self._setup_embeddings()
        
        # Initialize FAISS vector store
        self.vector_store_manager = VectorStoreManager(
            vector_db_path=vector_db_path,
            collection_name=collection_name,
        )
        
        self.pdf_loader = PDFLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        self.retriever = Retriever(
            index=self.vector_store_manager.get_index(),
            top_k=top_k,
        )
        
        logger.info("   ✅ RAG service initialized")
        logger.info("="*70)
    
    def _setup_embeddings(self) -> None:
        """Setup HuggingFace embedding model."""
        logger.info("   Setting up embedding model...")
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("   ✅ Using HuggingFace embeddings (all-MiniLM-L6-v2)")
        except ImportError:
            logger.warning("   ⚠️  HuggingFace embeddings not available")
            raise ImportError(
                "HuggingFace embeddings required. Install with: pip install llama-index-embeddings-huggingface"
            )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a PDF file and return documents.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        return self.pdf_loader.load_pdf(pdf_path)
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
        """
        logger.info("="*70)
        logger.info("RAG SERVICE: ADDING DOCUMENTS")
        logger.info("="*70)
        logger.info(f"   Number of documents: {len(documents)}")
        
        try:
            # Get index
            index = self.vector_store_manager.get_index()
            
            # Chunk documents first to see chunking info
            from llama_index.core import Settings
            node_parser = Settings.node_parser
            
            total_chunks = 0
            for i, doc in enumerate(documents):
                logger.info(f"   Processing document {i+1}/{len(documents)}...")
                # Get nodes (chunks) from document
                nodes = node_parser.get_nodes_from_documents([doc])
                total_chunks += len(nodes)
                logger.info(f"      Created {len(nodes)} chunks from document {i+1}")
                
                # Insert document (will use the chunks)
                index.insert(doc)
            
            logger.info(f"   ✅ Total chunks created: {total_chunks}")
            
            # Persist vector store
            self.vector_store_manager.persist()
            
            logger.info(f"   ✅ Successfully added {len(documents)} document(s) to vector store")
            
            # Get collection count (should be number of chunks/nodes)
            count = self.vector_store_manager.get_document_count()
            logger.info(f"   Total chunks in collection: {count}")
            logger.info("="*70)
        except Exception as e:
            logger.error(f"   ❌ ERROR adding documents: {e}", exc_info=True)
            raise
    
    def load_and_index_pdf(self, pdf_path: str) -> None:
        """
        Load a PDF and add it to the vector store in one step.
        
        Args:
            pdf_path: Path to the PDF file
        """
        logger.info("="*70)
        logger.info("RAG SERVICE: LOAD AND INDEX PDF")
        logger.info("="*70)
        
        documents = self.load_pdf(pdf_path)
        self.add_documents(documents)
        
        logger.info("="*70)
    
    def query(self, query_text: str, top_k: Optional[int] = None) -> str:
        """
        Query the vector store and return relevant context.
        
        Args:
            query_text: The query/question to search for
            top_k: Number of top results to retrieve (overrides default)
            
        Returns:
            Combined context from retrieved documents
        """
        return self.retriever.query(query_text, top_k=top_k)
    
    def get_retriever(self, top_k: Optional[int] = None):
        """
        Get a retriever object for more advanced retrieval.
        
        Args:
            top_k: Number of top results to retrieve
            
        Returns:
            Retriever object
        """
        return self.retriever.get_retriever(top_k=top_k)
    
    def clear_collection(self) -> None:
        """
        Clear all documents from the collection.
        """
        self.vector_store_manager.clear()
    
    def get_collection_count(self) -> int:
        """
        Get the number of documents in the collection.
        
        Returns:
            Number of documents
        """
        return self.vector_store_manager.get_document_count()


# Global RAG service instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """
    Get or create the global RAG service instance.
    Uses default settings optimized for FAISS semantic search.
    
    Returns:
        RAGService instance
    """
    global _rag_service
    
    if _rag_service is None:
        # Default settings - no config needed, RAG is always enabled
        _rag_service = RAGService(
            vector_db_path=None,  # Uses default ./vector_db
            collection_name="voice_assistant_rag",
            chunk_size=200,
            chunk_overlap=20,
            top_k=3,
        )
    
    return _rag_service

