"""
Retrieval Module
Handles querying and retrieving relevant context from vector store
"""
import logging
from typing import List, Optional
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import MetadataFilters

logger = logging.getLogger(__name__)


class Retriever:
    """
    Handles retrieval of relevant context from vector store.
    """
    
    def __init__(
        self,
        index: VectorStoreIndex,
        top_k: int = 3,
        metadata_filters: Optional[MetadataFilters] = None,
    ):
        """
        Initialize retriever.
        
        Args:
            index: VectorStoreIndex to query
            top_k: Number of top results to retrieve
        """
        self.index = index
        self.top_k = top_k
        self.query_engine = None
        self.metadata_filters = metadata_filters
        self._oversample_multiplier = 3
        self._oversample_min = 10
    
    def query(self, query_text: str, top_k: Optional[int] = None, use_llm: bool = False, llm=None) -> str:
        """
        Query the vector store and return relevant context.
        
        Args:
            query_text: The query/question to search for
            top_k: Number of top results to retrieve (overrides default)
            use_llm: Whether to use LLM to generate response (default: False, just returns chunks)
            llm: Optional LLM instance to use for response generation
            
        Returns:
            Combined context from retrieved documents (or LLM response if use_llm=True)
        """
        logger.info("="*70)
        logger.info("RETRIEVER: QUERY")
        logger.info("="*70)
        logger.info(f"   Query: {query_text}")
        
        k = top_k if top_k is not None else self.top_k
        logger.info(f"   Top K: {k}")
        logger.info(f"   Use LLM: {use_llm}")
        
        try:
            if use_llm and llm:
                # Use LLM to generate response from retrieved context
                if self.query_engine is None:
                    self.query_engine = self.index.as_query_engine(
                        similarity_top_k=k,
                        response_mode="compact",
                        llm=llm,
                    )
                
                response = self.query_engine.query(query_text)
                
                # Extract source nodes for context
                context_parts = []
                if hasattr(response, 'source_nodes') and response.source_nodes:
                    for i, node in enumerate(response.source_nodes[:k]):
                        node_text = node.node.text if hasattr(node, 'node') else node.text
                        context_parts.append(f"[Context {i+1}]: {node_text}")
                        logger.info(f"   Retrieved context {i+1}: {len(node_text)} chars")
                
                context = "\n\n".join(context_parts) if context_parts else ""
                llm_response = str(response) if hasattr(response, '__str__') else ""
                
                logger.info(f"   ✅ Query completed, retrieved {len(context_parts)} context(s)")
                logger.info(f"   Total context length: {len(context)} chars")
                logger.info("="*70)
                
                return f"LLM Response: {llm_response}\n\nRetrieved Context:\n{context}"
            else:
                # Just retrieve chunks without LLM
                query_bundle = QueryBundle(query_text)
                nodes = self._retrieve_nodes_with_filters(query_bundle, k)
                
                context_parts = []
                for i, node in enumerate(nodes):
                    # NodeWithScore has .node attribute containing the actual node
                    node_text = node.node.text if hasattr(node, 'node') else node.text
                    context_parts.append(f"[Context {i+1}]: {node_text}")
                    logger.info(f"   Retrieved context {i+1}: {len(node_text)} chars")
                
                context = "\n\n".join(context_parts) if context_parts else ""
                
                logger.info(f"   ✅ Query completed, retrieved {len(context_parts)} context(s)")
                logger.info(f"   Total context length: {len(context)} chars")
                logger.info("="*70)
                
                return context
        except Exception as e:
            logger.error(f"   ❌ ERROR querying: {e}", exc_info=True)
            raise
    
    def retrieve_nodes(self, query_text: str, top_k: Optional[int] = None) -> List[NodeWithScore]:
        """
        Retrieve nodes directly (for more advanced use cases).
        
        Args:
            query_text: The query/question to search for
            top_k: Number of top results to retrieve
            
        Returns:
            List of NodeWithScore objects
        """
        k = top_k if top_k is not None else self.top_k
        
        try:
            query_bundle = QueryBundle(query_text)
            nodes = self._retrieve_nodes_with_filters(query_bundle, k)
            return nodes
        except Exception as e:
            logger.error(f"   ❌ ERROR retrieving nodes: {e}", exc_info=True)
            raise
    
    def get_retriever(self, top_k: Optional[int] = None):
        """
        Get a retriever object for more advanced retrieval.
        
        Args:
            top_k: Number of top results to retrieve
            
        Returns:
            Retriever object
        """
        k = top_k if top_k is not None else self.top_k
        return self.index.as_retriever(similarity_top_k=k)
    
    def get_query_engine(self, top_k: Optional[int] = None):
        """
        Get a query engine for querying.
        
        Args:
            top_k: Number of top results to retrieve
            
        Returns:
            QueryEngine object
        """
        k = top_k if top_k is not None else self.top_k
        return self.index.as_query_engine(similarity_top_k=k)

    def _retrieve_nodes_with_filters(
        self, query_bundle: QueryBundle, top_k: int
    ) -> List[NodeWithScore]:
        """
        Retrieve nodes and apply metadata filters manually if needed.
        """
        similarity_top_k = self._determine_similarity_top_k(top_k)
        retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)
        nodes = retriever.retrieve(query_bundle)
        filtered_nodes = self._apply_metadata_filters(nodes)

        if self.metadata_filters:
            if len(filtered_nodes) < len(nodes):
                logger.debug(
                    "   Metadata filters removed %d of %d nodes",
                    len(nodes) - len(filtered_nodes),
                    len(nodes),
                )
            if len(filtered_nodes) < top_k:
                logger.debug(
                    "   Only %d node(s) left after filtering; requested top_k=%d",
                    len(filtered_nodes),
                    top_k,
                )

        return filtered_nodes[:top_k]

    def _determine_similarity_top_k(self, requested_top_k: int) -> int:
        """
        Determine how many candidates to request from the vector store.
        Oversample when metadata filters are present so we still have enough
        results after manual filtering.
        """
        if not self.metadata_filters:
            return requested_top_k
        oversampled = max(
            requested_top_k * self._oversample_multiplier,
            requested_top_k + self._oversample_min,
        )
        return max(oversampled, requested_top_k)

    def _apply_metadata_filters(
        self, nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """
        Apply simple equality-based metadata filters manually.
        """
        if (
            not self.metadata_filters
            or not getattr(self.metadata_filters, "filters", None)
        ):
            return nodes

        filtered_nodes: List[NodeWithScore] = []
        for node in nodes:
            node_metadata = {}
            if hasattr(node, "node") and node.node is not None:
                node_metadata = dict(getattr(node.node, "metadata", {}) or {})
            else:
                node_metadata = dict(getattr(node, "metadata", {}) or {})

            if self._metadata_matches(node_metadata):
                filtered_nodes.append(node)

        return filtered_nodes

    def _metadata_matches(self, metadata: dict) -> bool:
        """
        Check whether metadata satisfies all configured filters.
        Currently supports equality checks, which is our only use-case.
        """
        if not self.metadata_filters:
            return True

        for meta_filter in getattr(self.metadata_filters, "filters", []):
            key = getattr(meta_filter, "key", None)
            expected_value = getattr(meta_filter, "value", None)
            operator = getattr(meta_filter, "operator", None)

            # Treat None or 'eq'/'==' as equality checks.
            if operator not in (None, "eq", "=="):
                logger.debug(
                    "   Unsupported metadata filter operator '%s' for key '%s'; ignoring filter.",
                    operator,
                    key,
                )
                continue

            if metadata.get(key) != expected_value:
                return False

        return True

