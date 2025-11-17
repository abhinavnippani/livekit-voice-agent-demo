"""
Orchestrator - Multi-Agent Routing and Handoff Management
Manages conversation flow between specialized person agents
"""

import logging
import random
from typing import Optional, List, Dict
from llama_index.core import VectorStoreIndex

from .person_agent import PersonAgent, create_person_agent, PERSON_CONFIGS

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrates conversations between multiple person agents.
    Handles initial routing, topic detection, and agent handoffs.
    """
    
    def __init__(
        self,
        person_agents: List[PersonAgent],
        current_person: Optional[PersonAgent] = None
    ):
        """
        Initialize orchestrator with person agents.
        
        Args:
            person_agents: List of PersonAgent instances
            current_person: Currently active person (None = random selection)
        """
        self.person_agents = person_agents
        self.current_person = current_person or random.choice(person_agents)
        self.conversation_history: List[Dict] = []
        self.met_people = {self.current_person.name}
        
        # Create topic to agent mapping
        self.topic_to_agent: Dict[str, PersonAgent] = {
            agent.topic: agent for agent in person_agents
        }
        
        logger.info("="*70)
        logger.info("ORCHESTRATOR: INITIALIZATION")
        logger.info("="*70)
        logger.info(f"   Number of agents: {len(person_agents)}")
        logger.info(f"   Initial person: {self.current_person.name} ({self.current_person.topic})")
        for agent in person_agents:
            logger.info(f"   - {agent.name}: {agent.topic} ({agent.personality_type})")
        logger.info("="*70)
    
    def get_initial_greeting(self) -> str:
        """
        Get initial greeting from the randomly selected person.
        
        Returns:
            Greeting message from current person
        """
        return self.current_person.get_greeting()
    
    def detect_topic(self, query: str) -> Optional[str]:
        """
        Detect which topic a query is about using simple keyword matching.
        Could be enhanced with embeddings or LLM classification.
        
        Args:
            query: User's question
        
        Returns:
            Detected topic or None
        """
        query_lower = query.lower()
        
        # Special case: user explicitly asks for another person
        special_person = self._detect_person_request(query_lower)
        if special_person:
            return special_person.topic
        
        # Check each topic for keyword matches
        topic_scores = {}
        for topic in self.topic_to_agent.keys():
            # Split topic into keywords (e.g., "artificial_intelligence" -> ["artificial", "intelligence"])
            keywords = topic.replace("_", " ").split()
            
            # Count matches
            score = sum(1 for keyword in keywords if keyword in query_lower)
            
            # Add topic-specific keyword matching
            if topic == "interruption" and any(
                keyword in query_lower
                for keyword in [
                    "interrupt",
                    "interruption",
                    "barge-in",
                    "barge in",
                    "bargein",
                    "turn-taking",
                    "turn taking",
                    "talk over",
                    "double talk",
                    "overlap speech",
                ]
            ):
                score += 2
            elif topic == "latency" and any(
                keyword in query_lower
                for keyword in [
                    "latency",
                    "lag",
                    "delay",
                    "round trip",
                    "rtt",
                    "milliseconds",
                    "jitter",
                    "throughput",
                ]
            ):
                score += 2
            elif topic == "streaming" and any(
                keyword in query_lower
                for keyword in [
                    "streaming",
                    "live audio",
                    "live stream",
                    "buffer",
                    "bitrate",
                    "rtmp",
                    "rtp",
                    "chunked",
                    "pipeline",
                    "real-time",
                ]
            ):
                score += 2
            
            if score > 0:
                topic_scores[topic] = score
        
        # Return topic with highest score
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            logger.info(f"   Detected topic: {best_topic} (score: {topic_scores[best_topic]})")
            return best_topic
        
        return None
    
    def should_handoff(self, query: str) -> Optional[PersonAgent]:
        """
        Determine if query should be handed off to a different person.
        
        Args:
            query: User's question
        
        Returns:
            PersonAgent to hand off to, or None if current person should handle it
        """
        detected_topic = self.detect_topic(query)
        
        if detected_topic is None:
            # No clear topic detected, current person handles it
            return None
        
        if detected_topic == self.current_person.topic:
            # Current person's topic
            return None
        
        # Different topic detected - handoff needed
        target_agent = self.topic_to_agent.get(detected_topic)
        if target_agent:
            logger.info(f"   🔄 Handoff detected: {self.current_person.name} -> {target_agent.name}")
            return target_agent
        
        return None
    
    def handle_query(self, query: str) -> Dict[str, any]:
        """
        Main query handler that orchestrates the conversation.
        
        Args:
            query: User's question
        
        Returns:
            Dict with response information including context, person, and handoff status
        """
        logger.info("="*70)
        logger.info("ORCHESTRATOR: HANDLE QUERY")
        logger.info("="*70)
        logger.info(f"   Current person: {self.current_person.name}")
        logger.info(f"   Query: {query[:100]}...")
        
        query_lower = query.lower()
        requested_person = self._detect_person_request(query_lower)
        
        if requested_person:
            return self._perform_handoff(requested_person, query, force_handoff_message=True)
        
        # Check if handoff is needed based on topic
        handoff_to = self.should_handoff(query)
        
        if handoff_to and handoff_to != self.current_person:
            return self._perform_handoff(handoff_to, query)
        
        # Current person handles the query
        logger.info(f"   {self.current_person.name} handling query")
        
        # Retrieve persona context + knowledge chunks from current person's knowledge base
        context_bundle = self.current_person.query(query)
        combined_context = context_bundle.get("combined_context", "")
        
        # Log conversation history
        self.conversation_history.append({
            "query": query,
            "person": self.current_person.name,
            "handoff": False
        })
        
        self.met_people.add(self.current_person.name)
        system_prompt = self.current_person.get_system_prompt(
            peers=self._get_peer_descriptions(self.current_person),
            conversation_summary=self.get_conversation_summary(),
        )
        result = {
            "person": self.current_person.name,
            "person_obj": self.current_person,
            "handoff": False,
            "handoff_message": None,
            "context": combined_context,
            "context_chunks": context_bundle.get("chunks", []),
            "persona_context": context_bundle.get("persona_context", ""),
            "chunk_block": context_bundle.get("chunk_block", ""),
            "system_prompt": system_prompt,
            "topic": self.current_person.topic
        }
        
        logger.info("   ✅ Query handled")
        logger.info("="*70)
        return result
    
    def get_current_person(self) -> PersonAgent:
        """Get currently active person."""
        return self.current_person
    
    def set_current_person(self, person: PersonAgent) -> None:
        """Set currently active person."""
        self.current_person = person
        logger.info(f"   Current person set to: {person.name}")
    
    def get_all_people(self) -> List[PersonAgent]:
        """Get list of all person agents."""
        return self.person_agents
    
    def get_conversation_summary(self) -> str:
        """
        Get summary of conversation history.
        
        Returns:
            Formatted conversation summary
        """
        if not self.conversation_history:
            return "No conversation history yet."
        
        summary = "Conversation History:\n"
        for i, entry in enumerate(self.conversation_history, 1):
            if entry.get("handoff"):
                summary += f"{i}. Handoff: {entry['from_person']} -> {entry['to_person']}\n"
            else:
                summary += f"{i}. Query handled by {entry['person']}\n"
        
        return summary
    
    def _perform_handoff(
        self,
        handoff_to: PersonAgent,
        query: str,
        force_handoff_message: bool = False
    ) -> Dict[str, any]:
        logger.info(f"   Handoff: {self.current_person.name} -> {handoff_to.name}")
        
        handoff_message = ""
        if handoff_to == self.current_person and not force_handoff_message:
            logger.info("   Handoff target is current person; skipping.")
            return {
                "person": self.current_person.name,
                "person_obj": self.current_person,
                "handoff": False,
                "handoff_message": None,
                "context": "",
                "system_prompt": self.current_person.get_system_prompt(
                    peers=self._get_peer_descriptions(self.current_person),
                    conversation_summary=self.get_conversation_summary(),
                ),
                "topic": self.current_person.topic,
            }

        if force_handoff_message or handoff_to != self.current_person:
            readable_topic = handoff_to.topic.replace("_", " ")
            handoff_message = (
                f"That's not really my area, but {handoff_to.name} can help you with {readable_topic}.\n"
                f"{handoff_to.get_greeting()}"
            )
        
        old_person = self.current_person
        self.current_person = handoff_to
        
        self.conversation_history.append({
            "query": query,
            "from_person": old_person.name,
            "to_person": handoff_to.name,
            "handoff": True
        })
        
        system_prompt = handoff_to.get_system_prompt(
            peers=self._get_peer_descriptions(handoff_to),
            conversation_summary=self.get_conversation_summary(),
        )
        self.met_people.add(handoff_to.name)
        
        result = {
            "person": handoff_to.name,
            "person_obj": handoff_to,
            "handoff": True,
            "handoff_message": handoff_message,
            "context": "",
            "system_prompt": system_prompt,
            "topic": handoff_to.topic
        }
        
        logger.info("   ✅ Handoff completed")
        logger.info("="*70)
        return result

    def _get_peer_descriptions(self, exclude_person: PersonAgent) -> List[str]:
        peers = []
        for agent in self.person_agents:
            if agent != exclude_person:
                peers.append(f"{agent.name} who specializes in {agent.topic.replace('_', ' ')}")
        return peers

    def _detect_person_request(self, query_lower: str) -> Optional[PersonAgent]:
        named_person = self._detect_named_person_request(query_lower)
        if named_person:
            return named_person
        
        keywords = [
            "other person",
            "someone else",
            "another expert",
            "third person",
            "other expert",
        ]
        if not any(k in query_lower for k in keywords):
            return None
        
        for agent in self.person_agents:
            if agent.name not in self.met_people:
                return agent
        for agent in self.person_agents:
            if agent != self.current_person:
                return agent
        return None

    def _detect_named_person_request(self, query_lower: str) -> Optional[PersonAgent]:
        if not self._contains_person_request_trigger(query_lower):
            return None
        
        for agent in self.person_agents:
            variants = self._get_agent_name_variants(agent)
            if any(variant in query_lower for variant in variants):
                return agent
        return None

    def _contains_person_request_trigger(self, query_lower: str) -> bool:
        triggers = [
            "connect",
            "introduce",
            "talk to",
            "talk with",
            "speak to",
            "speak with",
            "let me talk",
            "let me speak",
            "bring in",
            "bring over",
            "switch to",
            "hand off",
            "handoff",
            "hand me",
            "meet",
            "hear from",
            "get",
        ]
        return any(trigger in query_lower for trigger in triggers)

    def _get_agent_name_variants(self, agent: PersonAgent) -> set[str]:
        name_lower = agent.name.lower()
        parts = [part for part in name_lower.split() if part]
        variants = {name_lower}
        variants.update(parts)
        return variants


def create_orchestrator(
    topics: List[str],
    indices: Dict[str, VectorStoreIndex],
    top_k: int = 3
) -> Orchestrator:
    """
    Factory function to create an Orchestrator with person agents.
    
    Args:
        topics: List of topics (must match predefined PERSON_CONFIGS)
        indices: Dict mapping topic to VectorStoreIndex
        top_k: Number of chunks to retrieve per query
    
    Returns:
        Orchestrator instance
    """
    logger.info("Creating orchestrator with person agents...")
    
    person_agents = []
    for topic in topics:
        if topic not in indices:
            logger.warning(f"⚠️  No index found for topic: {topic}, skipping")
            continue
        
        try:
            agent = create_person_agent(
                topic=topic,
                index=indices[topic],
                top_k=top_k
            )
            person_agents.append(agent)
        except ValueError as e:
            logger.error(f"❌ Error creating agent for {topic}: {e}")
    
    if not person_agents:
        raise ValueError("No person agents created. Check topics and indices.")
    
    # Randomly select initial person
    orchestrator = Orchestrator(person_agents=person_agents)
    
    logger.info(f"✅ Orchestrator created with {len(person_agents)} agents")
    return orchestrator