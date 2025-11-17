"""
Multi-Agent RAG Service
Orchestrates multiple specialized agents with topic-specific knowledge bases
"""

import logging
import os
from typing import Optional, List, Dict
from pathlib import Path
from llama_index.core import Document, Settings

from .single_faiss_multi_collection import SingleFAISSMultiCollection
from .pdf_loader import PDFLoader
from .orchestrator import Orchestrator, create_orchestrator
from .person_agent import PERSON_CONFIGS

logger = logging.getLogger(__name__)


class MultiAgentRAGService:
    """
    Multi-Agent RAG service that manages multiple specialized person agents.
    Each agent has their own FAISS collection with topic-specific knowledge.
    """
    
    def __init__(
        self,
        vector_db_path: Optional[str] = None,
        topics: Optional[List[str]] = None,
        chunk_size: int = 200,
        chunk_overlap: int = 20,
        top_k: int = 3,
    ):
        """
        Initialize multi-agent RAG service.
        
        Args:
            vector_db_path: Path to store vector databases
            topics: List of topics (one collection per topic)
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
            top_k: Number of top results to retrieve per query
        """
        logger.info("="*70)
        logger.info("MULTI-AGENT RAG SERVICE: INITIALIZATION")
        logger.info("="*70)
        
        # Default topics from predefined person configs
        if topics is None:
            topics = [config["topic"] for config in PERSON_CONFIGS]
        
        self.topics = topics
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
        # Initialize embedding model
        self._setup_embeddings()
        
        # Initialize multi-collection vector store
        self.vector_store = SingleFAISSMultiCollection(
            vector_db_path=vector_db_path,
            topics=topics,
        )
        
        # Initialize PDF loader
        self.pdf_loader = PDFLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # Initialize orchestrator with all agents
        indices = self.vector_store.get_all_indices()
        self.orchestrator = create_orchestrator(
            topics=topics,
            indices=indices,
            top_k=top_k
        )
        
        logger.info("   ✅ Multi-agent RAG service initialized")
        logger.info(f"   Topics: {', '.join(topics)}")
        logger.info(f"   Agents: {len(self.orchestrator.get_all_people())}")
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
            logger.warning("   ⚠️ HuggingFace embeddings not available")
            raise ImportError(
                "HuggingFace embeddings required. Install with: pip install llama-index-embeddings-huggingface"
            )
    
    def load_pdf_for_topic(self, pdf_path: str, topic: str) -> None:
        """
        Load a PDF file and add it to a specific topic's collection.
        
        Args:
            pdf_path: Path to the PDF file
            topic: Topic to add the PDF to
        """
        logger.info("="*70)
        logger.info(f"MULTI-AGENT RAG: LOAD PDF FOR TOPIC: {topic}")
        logger.info("="*70)
        logger.info(f"   PDF: {pdf_path}")
        
        if topic not in self.topics:
            raise ValueError(f"Topic '{topic}' not found. Available topics: {self.topics}")
        
        # Load PDF
        documents = self.pdf_loader.load_pdf(pdf_path)
        
        # Get index for this topic
        store = self.vector_store.get_store(topic)
        if not store:
            raise ValueError(f"No vector store found for topic: {topic}")
        
        index = store.get_index()
        
        # Add documents to topic-specific index
        logger.info(f"   Adding documents to {topic} collection...")
        
        from llama_index.core import Settings
        node_parser = Settings.node_parser
        
        total_chunks = 0
        for i, doc in enumerate(documents):
            logger.info(f"   Processing document {i+1}/{len(documents)}...")
            nodes = node_parser.get_nodes_from_documents([doc])
            for node in nodes:
                node.metadata = dict(node.metadata or {})
                node.metadata["topic"] = topic
            total_chunks += len(nodes)
            logger.info(f"      Created {len(nodes)} chunks from document {i+1}")
            if nodes:
                index.insert_nodes(nodes)
        
        logger.info(f"   ✅ Total chunks created: {total_chunks}")
        
        # Persist
        store.persist()
        
        count = store.get_document_count()
        logger.info(f"   ✅ Successfully added PDF to {topic}")
        logger.info(f"   Total chunks in {topic} collection: {count}")
        logger.info("="*70)
    
    def query(self, query_text: str) -> Dict[str, any]:
        """
        Query the multi-agent system.
        The orchestrator will route to the appropriate person agent.
        
        Args:
            query_text: The user's question
        
        Returns:
            Dict containing response context, person info, and handoff status
        """
        return self.orchestrator.handle_query(query_text)
    
    def get_initial_greeting(self) -> str:
        """
        Get initial greeting from randomly selected person.
        
        Returns:
            Greeting message
        """
        return self.orchestrator.get_initial_greeting()
    
    def get_current_person(self):
        """Get currently active person agent."""
        return self.orchestrator.get_current_person()
    
    def get_all_people(self):
        """Get all person agents."""
        return self.orchestrator.get_all_people()
    
    def get_collection_counts(self) -> Dict[str, int]:
        """
        Get document counts for all topic collections.
        
        Returns:
            Dict mapping topic to document count
        """
        return self.vector_store.get_all_counts()
    
    def clear_topic(self, topic: str) -> None:
        """
        Clear all documents from a specific topic collection.
        
        Args:
            topic: Topic to clear
        """
        store = self.vector_store.get_store(topic)
        if store:
            store.clear()
            logger.info(f"✅ Cleared collection for topic: {topic}")
        else:
            logger.warning(f"⚠️ Topic not found: {topic}")
    
    def clear_all(self) -> None:
        """Clear all topic collections."""
        self.vector_store.clear_all()
        logger.info("✅ Cleared all collections")
    
    def get_conversation_summary(self) -> str:
        """Get summary of conversation history."""
        return self.orchestrator.get_conversation_summary()
    
    def get_topics(self) -> List[str]:
        """Get list of all topics."""
        return self.topics.copy()


# Global multi-agent RAG service instance
_multi_agent_rag_service: Optional[MultiAgentRAGService] = None


def get_multi_agent_rag_service(
    topics: Optional[List[str]] = None,
    vector_db_path: Optional[str] = None,
) -> MultiAgentRAGService:
    """
    Get or create the global multi-agent RAG service instance.
    
    Args:
        topics: List of topics (uses defaults if None)
        vector_db_path: Path to vector database
    
    Returns:
        MultiAgentRAGService instance
    """
    global _multi_agent_rag_service
    
    if _multi_agent_rag_service is None:
        # Use predefined topics from PERSON_CONFIGS if not provided
        if topics is None:
            topics = [config["topic"] for config in PERSON_CONFIGS]
        
        _multi_agent_rag_service = MultiAgentRAGService(
            vector_db_path=vector_db_path,
            topics=topics,
            chunk_size=200,
            chunk_overlap=20,
            top_k=3,
        )
    
    return _multi_agent_rag_service


def reset_multi_agent_rag_service():
    """Reset the global multi-agent RAG service (useful for testing)."""
    global _multi_agent_rag_service
    _multi_agent_rag_service = None