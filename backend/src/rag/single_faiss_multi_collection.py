"""
PROPER SOLUTION: Single FAISS Index with Metadata Filtering
This is the standard LlamaIndex pattern - much simpler and cleaner!
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    load_index_from_storage,
)
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

logger = logging.getLogger(__name__)


@dataclass
class TopicStore:
    """Lightweight helper exposing topic-scoped operations."""

    topic: str
    parent: "SingleIndexMultiTopic"

    def get_index(self) -> Optional[VectorStoreIndex]:
        """Get the shared index (same for all topics)."""
        return self.parent.index

    def add_documents(self, documents: List[Document]) -> None:
        self.parent.add_documents(self.topic, documents)

    def persist(self) -> None:
        self.parent.persist()

    def get_document_count(self) -> int:
        return self.parent.get_document_count(self.topic).get(self.topic, 0)

    def clear(self) -> None:
        self.parent.clear_topic(self.topic)


class SingleFAISSMultiCollection:
    """
    Single FAISS index with metadata filtering for multiple topics.
    This is the CORRECT and SIMPLE approach!
    
    Key points:
    - ONE FAISS index stores ALL documents
    - Each document has metadata: {"topic": "interruption"}
    - Filter at query time using MetadataFilters
    """
    
    def __init__(
        self,
        vector_db_path: Optional[str] = None,
        topics: Optional[List[str]] = None,
        embedding_dim: int = 384,
    ):
        """
        Initialize single FAISS index with metadata filtering.
        
        Args:
            vector_db_path: Path to store the FAISS index
            topics: List of topic names
            embedding_dim: Embedding dimension (384 for all-MiniLM-L6-v2)
        """
        self.vector_db_path = Path(vector_db_path or "./vector_db")
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        self.topics = topics or []
        self.embedding_dim = embedding_dim
        
        logger.info("="*70)
        logger.info("SINGLE INDEX WITH METADATA FILTERING: INITIALIZATION")
        logger.info("="*70)
        logger.info(f"   Path: {self.vector_db_path}")
        logger.info(f"   Topics: {self.topics}")
        logger.info(f"   Embedding dim: {embedding_dim}")
        
        # Initialize components
        self.faiss_index = self._init_faiss_index()
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        self.storage_context = self._init_storage_context()
        self.index = self._init_index()
        
        logger.info("   ✅ Single FAISS index with metadata filtering initialized")
        logger.info("="*70)
    
    def _init_faiss_index(self) -> faiss.Index:
        """Initialize or load the FAISS index."""
        index_path = self.vector_db_path / "faiss_index.index"
        
        if index_path.exists():
            try:
                logger.info(f"   Loading existing FAISS index from {index_path}")
                faiss_index = faiss.read_index(str(index_path))
                logger.info(f"   ✅ Loaded index with {faiss_index.ntotal} vectors")
                return faiss_index
            except Exception as e:
                logger.warning(f"   ⚠️ Could not load index: {e}, creating new")
        
        # Create new FAISS index
        logger.info(f"   Creating new FAISS index (dim={self.embedding_dim})")
        faiss_index = faiss.IndexFlatL2(self.embedding_dim)
        return faiss_index
    
    def _init_storage_context(self) -> StorageContext:
        """Load existing storage context if available."""
        persist_dir = self.vector_db_path
        if persist_dir.exists() and any(persist_dir.iterdir()):
            try:
                logger.info(f"   Loading storage context from {persist_dir}")
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(persist_dir),
                    vector_store=self.vector_store,
                )
                logger.info(f"   ✅ Storage context loaded with {len(storage_context.docstore.docs)} documents")
                return storage_context
            except Exception as e:
                logger.warning(f"   ⚠️ Could not load storage context: {e}")
        
        return StorageContext.from_defaults(vector_store=self.vector_store)
    
    def _init_index(self) -> VectorStoreIndex:
        """Initialize or load the vector index."""
        persist_dir = self.vector_db_path
        
        if persist_dir.exists() and any(persist_dir.iterdir()):
            try:
                logger.info(f"   Loading existing index from {persist_dir}")
                index = load_index_from_storage(storage_context=self.storage_context)
                logger.info(f"   ✅ Loaded existing index")
                return index
            except Exception as e:
                logger.warning(f"   ⚠️ Could not load index: {e}, creating new")
        
        # Create new index
        logger.info("   Creating new index")
        index = VectorStoreIndex([], storage_context=self.storage_context)
        return index
    
    def get_index(self, topic: str) -> Optional[VectorStoreIndex]:
        """
        Get index for a specific topic (same index for all, filtered at query time).
        
        Args:
            topic: Topic name
        
        Returns:
            The shared VectorStoreIndex
        """
        if topic not in self.topics:
            return None
        return self.index
    
    def add_documents(self, topic: str, documents: List[Document]) -> None:
        """
        Add documents to the index with topic metadata.
        
        Args:
            topic: Topic name
            documents: Documents to add
        """
        if topic not in self.topics:
            raise ValueError(f"Topic '{topic}' not found. Available: {self.topics}")
        
        logger.info(f"Adding {len(documents)} documents to topic: {topic}")
        
        # Add topic metadata to all documents
        for doc in documents:
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["topic"] = topic
            self.index.insert(doc)
        
        logger.info(f"✅ Added documents to {topic}")
    
    def persist(self) -> None:
        """Persist the FAISS index to disk."""
        index_path = self.vector_db_path / "faiss_index.index"
        
        logger.info(f"Persisting FAISS index to {index_path}")
        
        # Write the FAISS index
        faiss.write_index(self.vector_store._faiss_index, str(index_path))
        
        # Persist storage context
        self.storage_context.persist(persist_dir=str(self.vector_db_path))
        
        logger.info("✅ FAISS index persisted")
    
    def get_document_count(self, topic: Optional[str] = None) -> Dict[str, int]:
        """
        Get document counts per topic.
        
        Args:
            topic: If specified, return count for that topic only
        
        Returns:
            Dict of topic -> count
        """
        docstore = self.storage_context.docstore
        
        if topic:
            count = sum(
                1 for doc_id, doc in docstore.docs.items()
                if doc.metadata.get("topic") == topic
            )
            return {topic: count}
        
        # Get counts for all topics
        counts = {}
        for t in self.topics:
            count = sum(
                1 for doc_id, doc in docstore.docs.items()
                if doc.metadata.get("topic") == t
            )
            counts[t] = count
        
        return counts
    
    def get_total_vectors(self) -> int:
        """Get total number of vectors in the FAISS index."""
        return self.vector_store._faiss_index.ntotal
    
    def clear_topic(self, topic: str) -> None:
        """
        Clear all documents for a specific topic.
        
        Args:
            topic: Topic to clear
        """
        logger.info(f"Clearing topic: {topic}")
        
        docstore = self.storage_context.docstore
        docs_to_remove = [
            doc_id for doc_id, doc in docstore.docs.items()
            if doc.metadata.get("topic") == topic
        ]
        
        for doc_id in docs_to_remove:
            del docstore.docs[doc_id]
        
        logger.info(f"✅ Cleared {len(docs_to_remove)} documents from {topic}")
    
    def clear_all(self) -> None:
        """Clear all documents."""
        for topic in self.topics:
            self.clear_topic(topic)
    
    def get_store(self, topic: str) -> Optional[TopicStore]:
        """Return a helper wrapper for the requested topic."""
        if topic not in self.topics:
            logger.warning(f"Requested unknown topic store: {topic}")
            return None
        return TopicStore(topic=topic, parent=self)
    
    def get_all_indices(self) -> Dict[str, VectorStoreIndex]:
        """
        Return mapping of topic -> VectorStoreIndex.
        Note: All topics use the same index, filtered at query time.
        """
        return {topic: self.index for topic in self.topics}
    
    def get_all_counts(self) -> Dict[str, int]:
        """Return document counts for all topics."""
        return self.get_document_count()