"""
FAISS Vector Store
High-performance vector database for semantic search
"""
import logging
import os
from pathlib import Path
from typing import Optional, Any
from llama_index.core import VectorStoreIndex, StorageContext

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages FAISS vector store initialization and operations.
    """
    
    def __init__(
        self,
        vector_db_path: Optional[str] = None,
        collection_name: str = "voice_assistant_rag",
    ):
        """
        Initialize FAISS vector store manager.
        
        Args:
            vector_db_path: Path to store vector database
            collection_name: Collection/namespace name
        """
        logger.info("="*70)
        logger.info("VECTOR STORE: INITIALIZATION (FAISS)")
        logger.info("="*70)
        
        self.collection_name = collection_name
        
        # Set default path
        if vector_db_path is None:
            vector_db_path = os.path.join(os.getcwd(), "vector_db")
        self.vector_db_path = Path(vector_db_path)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"   Vector DB path: {self.vector_db_path}")
        logger.info(f"   Collection name: {self.collection_name}")
        
        # Initialize FAISS vector store
        self.vector_store = self._create_faiss_store()
        
        # Check if we should load persisted storage context
        persist_path = self.vector_db_path / "faiss" / f"{self.collection_name}"
        if persist_path.exists() and any(persist_path.iterdir()):
            try:
                # Try to load persisted storage context (includes docstore and index structure)
                self.storage_context = StorageContext.from_defaults(
                    persist_dir=str(persist_path),
                    vector_store=self.vector_store  # Use the loaded FAISS vector store
                )
                logger.info("   ✅ Loaded persisted storage context")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not load persisted storage context: {e}")
                logger.warning(f"   Clearing FAISS index to prevent mismatch")
                # If storage context fails to load but vector store has data, clear everything
                # to prevent mismatch between embeddings and metadata
                import shutil
                if persist_path.exists():
                    shutil.rmtree(persist_path)
                # Recreate fresh vector store and storage context
                self.vector_store = self._create_faiss_store()
                self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        else:
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Create or load index
        self.index = self._create_index()
        
        logger.info("   ✅ FAISS vector store initialized")
        logger.info("="*70)
    
    def _create_faiss_store(self) -> Any:
        """Create FAISS vector store (faster, in-memory with optional persistence)."""
        logger.info("   Creating FAISS vector store...")
        try:
            from llama_index.vector_stores.faiss import FaissVectorStore
            import faiss
            
            # FAISS persist directory
            faiss_persist_dir = self.vector_db_path / "faiss"
            faiss_persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if persisted index exists
            persist_path = faiss_persist_dir / f"{self.collection_name}"
            
            if persist_path.exists() and any(persist_path.iterdir()):
                logger.info(f"   Loading existing FAISS index from {persist_path}")
                try:
                    # Try to load existing
                    vector_store = FaissVectorStore.from_persist_dir(
                        persist_dir=str(persist_path)
                    )
                    logger.info("   ✅ FAISS vector store loaded from disk")
                    return vector_store
                except Exception as e:
                    logger.warning(f"   ⚠️  Could not load existing index: {e}, creating new")
            
            logger.info("   Creating new FAISS index")
            # Create new FAISS index
            # Using dimension 384 for HuggingFace all-MiniLM-L6-v2 embeddings
            embedding_dimension = 384
            faiss_index = faiss.IndexFlatL2(embedding_dimension)
            
            # Create FAISS vector store with the index
            # Note: FaissVectorStore stores text by default in newer versions
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            
            logger.info(f"   ✅ FAISS vector store created with dimension {embedding_dimension} (HuggingFace)")
            return vector_store
        except ImportError as e:
            raise ImportError(
                f"FAISS not available: {e}. Install with: pip install faiss-cpu"
            )
    
    def _create_index(self) -> VectorStoreIndex:
        """Create or load vector store index."""
        # Check if index already exists in storage context
        persist_path = self.vector_db_path / "faiss" / f"{self.collection_name}"
        if persist_path.exists() and any(persist_path.iterdir()):
            try:
                # Try to load existing index from storage context
                from llama_index.core import load_index_from_storage
                index = load_index_from_storage(storage_context=self.storage_context)
                logger.info("   ✅ Loaded existing index from storage")
                return index
            except Exception as e:
                logger.warning(f"   ⚠️  Could not load existing index: {e}, creating new")
        
        # Create new index with empty nodes list and our storage context
        # The vector store will be used when documents are added
        index = VectorStoreIndex(
            nodes=[],
            storage_context=self.storage_context,
        )
        return index
    
    def persist(self) -> None:
        """Persist the FAISS vector store to disk."""
        try:
            # FAISS vector store persistence
            faiss_persist_dir = self.vector_db_path / "faiss"
            faiss_persist_dir.mkdir(parents=True, exist_ok=True)
            persist_path = faiss_persist_dir / self.collection_name
            
            # Persist through storage context (preferred method)
            if hasattr(self.index, 'storage_context'):
                self.index.storage_context.persist(persist_dir=str(persist_path))
            elif hasattr(self.index, 'persist'):
                # Try persisting the index directly
                self.index.persist(persist_path=str(persist_path))
            elif hasattr(self.vector_store, 'persist'):
                # Try vector store persist (may not have persist_dir parameter)
                try:
                    self.vector_store.persist()
                except TypeError:
                    # If persist() doesn't take parameters, try with path
                    try:
                        self.vector_store.persist(str(persist_path))
                    except:
                        pass  # Persistence is optional
            logger.info("   ✅ FAISS index persisted")
        except Exception as e:
            logger.warning(f"   ⚠️  Error persisting FAISS index: {e}")
    
    def get_index(self) -> VectorStoreIndex:
        """Get the vector store index."""
        return self.index
    
    def get_document_count(self) -> int:
        """Get the number of chunks/nodes in the FAISS vector store."""
        try:
            # Try to get count from FAISS index
            if hasattr(self.vector_store, 'index') and self.vector_store.index:
                return self.vector_store.index.ntotal
            
            # Try to get count from the VectorStoreIndex
            if hasattr(self.index, 'docstore') and hasattr(self.index.docstore, 'docs'):
                return len(self.index.docstore.docs)
            
            # Try to get count from storage context
            if hasattr(self.index, 'storage_context') and hasattr(self.index.storage_context, 'docstore'):
                if hasattr(self.index.storage_context.docstore, 'docs'):
                    return len(self.index.storage_context.docstore.docs)
            
            return 0
        except Exception:
            return 0
    
    def clear(self) -> None:
        """Clear all documents from the FAISS vector store."""
        logger.info("="*70)
        logger.info("VECTOR STORE: CLEARING")
        logger.info("="*70)
        
        try:
            # Delete FAISS index directory
            faiss_persist_dir = self.vector_db_path / "faiss" / self.collection_name
            if faiss_persist_dir.exists():
                import shutil
                shutil.rmtree(faiss_persist_dir)
            # Recreate vector store
            self.vector_store = self._create_faiss_store()
            self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index = self._create_index()
            
            logger.info(f"   ✅ FAISS vector store cleared")
            logger.info("="*70)
        except Exception as e:
            logger.error(f"   ❌ ERROR clearing vector store: {e}", exc_info=True)
            raise