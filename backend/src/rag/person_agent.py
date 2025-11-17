"""
Person Agent - Individual specialized agent with personality and backstory
Each person agent is an expert in one topic with a unique personality and background
"""

import logging
from typing import Optional, Dict, List, Any
from llama_index.core import VectorStoreIndex

from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

from .personalities import Personality, get_personality
from .retrieval import Retriever

logger = logging.getLogger(__name__)


class PersonAgent:
    """
    Represents a person at a networking event.
    Each person is an expert in one topic with a unique personality and backstory.
    """
    
    def __init__(
        self,
        name: str,
        topic: str,
        personality_type: str,
        backstory: str,
        index: VectorStoreIndex,
        top_k: int = 3,
    ):
        """
        Initialize a person agent.
        
        Args:
            name: Name of the person (e.g., "Dr. Sarah Chen")
            topic: Topic this person specializes in
            personality_type: Personality type (professional, enthusiastic, academic, casual, humorous)
            backstory: Background story and context for this person
            index: VectorStoreIndex for this person's topic
            top_k: Number of chunks to retrieve
        """
        self.name = name
        self.topic = topic
        self.personality_type = personality_type
        self.backstory = backstory
        self.personality: Personality = get_personality(personality_type)
        self.index = index
        self.top_k = top_k
        
        # Initialize retriever for this person's knowledge base
        topic_filter = MetadataFilters(
            filters=[MetadataFilter(key="topic", value=topic)]
        )
        self.retriever = Retriever(
            index=index,
            top_k=top_k,
            metadata_filters=topic_filter,
        )
        
        logger.info(f"✅ Created PersonAgent: {name} (Topic: {topic}, Personality: {personality_type})")
    
    def get_system_prompt(
        self,
        peers: Optional[list[str]] = None,
        conversation_summary: Optional[str] = None,
    ) -> str:
        """
        Generate system prompt for this person agent.
        
        Returns:
            Complete system prompt with personality, backstory, and topic specialization
        """
        base_prompt = f"""You are {self.name}, a person at a networking event.

BACKGROUND:
{self.backstory}

EXPERTISE:
You specialize in {self.topic}. This is your area of deep knowledge and passion.

PERSONALITY:
{self.personality.system_prompt_modifier}

IMPORTANT INSTRUCTIONS:
1. You ONLY answer questions related to {self.topic}. This is your area of expertise.
2. When answering questions about {self.topic}, you MUST use the context retrieved from your knowledge base to provide accurate, detailed information.
3. If someone asks about a topic you don't specialize in, respond naturally with something like:
   - "That's not really my area, but let me introduce you to [person name] who knows all about [their topic]!"
   - "I'm not the best person to answer that, but I know someone who is! You should talk to [person name] about [their topic]."
4. Stay in character with your {self.personality.name} personality and your background.
5. Be helpful, friendly, and authentic - you're at a networking event making connections.
6. Reference your background naturally when relevant to build rapport.
7. When you retrieve context from your knowledge base, integrate it smoothly into your response without explicitly saying "according to my knowledge base" or "based on the context".

Your response style: {self.personality.response_style}

Remember: You are {self.name}, and you're here to share your expertise in {self.topic} while helping people connect with the right experts!
"""
        extras = ""
        if peers:
            extras += "\nOTHER EXPERTS AT THE EVENT:\n"
            for peer in peers:
                extras += f"- {peer}\n"
        if conversation_summary and conversation_summary.strip() and conversation_summary.strip() != "No conversation history yet.":
            extras += f"\nCONVERSATION SUMMARY SO FAR:\n{conversation_summary}\n"
        return base_prompt + extras
    
    def get_greeting(self) -> str:
        """
        Get personalized greeting for this person.
        
        Returns:
            Greeting message with personality, name, and topic
        """
        greeting = self.personality.greeting_style.format(topic=self.topic)
        return f"Hi! I'm {self.name}. {greeting}"
    
    def get_introduction(self) -> str:
        """
        Get full introduction with backstory.
        
        Returns:
            Complete introduction including name, background, and topic
        """
        return f"""I'm {self.name}. {self.backstory}

My expertise is in {self.topic}, and I'd be happy to discuss it with you!"""
    
    def can_answer_topic(self, query: str) -> bool:
        """
        Determine if this person can answer a query based on their topic.
        This is a simple heuristic - in practice, the orchestrator handles routing.
        
        Args:
            query: User's question
        
        Returns:
            True if person can answer (topic-related), False otherwise
        """
        # Simple keyword matching - could be enhanced with embeddings
        topic_keywords = self.topic.lower().split()
        query_lower = query.lower()
        
        # Check if any topic keywords appear in query
        return any(keyword in query_lower for keyword in topic_keywords)
    
    def get_persona_profile(self) -> Dict[str, str]:
        """
        Return a structured description of this persona.
        """
        return {
            "name": self.name,
            "topic": self.topic,
            "personality": self.personality.name,
            "personality_type": self.personality_type,
            "personality_description": self.personality.description.strip(),
            "response_style": self.personality.response_style,
            "backstory": self.backstory.strip(),
        }
    
    def get_persona_context_block(self) -> str:
        """
        Render persona details as a block of text that can be embedded
        alongside retrieved knowledge chunks.
        """
        profile = self.get_persona_profile()
        lines = [
            "[Persona Overview]",
            f"Name: {profile['name']}",
            f"Topic: {profile['topic']}",
            f"Personality: {profile['personality']} ({profile['personality_type']})",
            f"Response Style: {profile['response_style']}",
            f"Summary: {profile['personality_description']}",
            "Backstory:",
            profile["backstory"],
        ]
        return "\n".join(line for line in lines if line).strip()
    
    def query(self, query_text: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Query this person's knowledge base and return both the retrieved
        chunks and persona information so downstream agents can stay
        persona-aware while responding.
        
        Args:
            query_text: Question to answer
            top_k: Number of chunks to retrieve (overrides default)
        
        Returns:
            Dict containing persona context, individual chunks, and the combined block
        """
        logger.info(f"   {self.name} querying knowledge base for: {query_text[:50]}...")
        nodes = self.retriever.retrieve_nodes(query_text, top_k=top_k)
        
        chunk_texts: List[str] = []
        formatted_chunks: List[str] = []
        for idx, node in enumerate(nodes, start=1):
            node_text = node.node.text if hasattr(node, 'node') and node.node is not None else getattr(node, 'text', "")
            clean_text = (node_text or "").strip()
            if not clean_text:
                continue
            chunk_texts.append(clean_text)
            formatted_chunks.append(f"[Context {idx}]: {clean_text}")
        
        chunk_block = "\n\n".join(formatted_chunks).strip()
        persona_block = self.get_persona_context_block()
        
        if chunk_block:
            combined_context = f"{persona_block}\n\n[Retrieved Knowledge]\n{chunk_block}"
        else:
            combined_context = persona_block
        
        return {
            "persona_context": persona_block,
            "chunks": chunk_texts,
            "chunk_block": chunk_block,
            "combined_context": combined_context,
        }
    
    def format_response_with_context(self, query: str, context: str) -> str:
        """
        Format a response that includes both the query context and personality.
        This is the context that will be given to the LLM.
        
        Args:
            query: User's question
            context: Retrieved context from knowledge base
        
        Returns:
            Formatted prompt with context
        """
        return f"""User asked: "{query}"

Retrieved Context from {self.name}'s Knowledge Base:
{context}

Please answer the user's question using the context above. Stay in character as {self.name} with a {self.personality.name} personality. If the context doesn't contain relevant information, be honest about it but stay helpful."""
    
    def __repr__(self) -> str:
        return f"PersonAgent(name={self.name}, topic={self.topic}, personality={self.personality_type})"


# Predefined person configurations with names and backstories
PERSON_CONFIGS = [
    {
        "name": "Skye Morales",
        "topic": "interruption",
        "personality": "comedian",
        "backstory": "I’m Skye Morales, FlowDial’s conversation design lead. I spent years stage-managing improv tours, so I physically cannot answer a question about interruptions without sneaking in a joke. My whole deal is choreographing barge-ins and double-talk so gracefully that the audience laughs instead of groans. If you’re talking overlapping speech, expect me to juggle examples, crack a punchline, and toss in a callback before the next beat hits.",
    },
    {
        "name": "Noah Reed",
        "topic": "latency",
        "personality": "professional",
        "backstory": "I’m Noah Reed, the network reliability lead for PulsePlay’s live esports broadcasts. I live in traceroutes and latency budgets, so my sentences tend to land crisp and efficient. My workbench is covered in packet captures, analog stopwatches, and fountain pens for the rare moments I write longhand. When I’m not shaving milliseconds off pipelines for voice agents, I train for century rides and foster senior greyhounds.",
    },
    {
        "name": "Avery Kim",
        "topic": "streaming",
        "personality": "aloof",
        "backstory": "I’m Avery Kim, slogging through a second-year HCII thesis on adaptive streaming. I keep the campus radio running because it pays for my synth habit, not because I enjoy small talk. Ask me about buffering or bitrates and I’ll answer—briefly. I’ll probably remind you I have better things to do, and if you expect warmth, maybe try the other experts.",
    }
]


def get_person_config(topic: str) -> Optional[dict]:
    """
    Get predefined person configuration for a topic.
    
    Args:
        topic: Topic name
    
    Returns:
        Person configuration dict or None
    """
    for config in PERSON_CONFIGS:
        if config["topic"] == topic:
            return config
    return None


def create_person_agent(
    topic: str,
    index: VectorStoreIndex,
    top_k: int = 3,
    custom_config: Optional[dict] = None
) -> PersonAgent:
    """
    Factory function to create a PersonAgent with predefined or custom configuration.
    
    Args:
        topic: Topic this person specializes in
        index: VectorStoreIndex for this topic
        top_k: Number of chunks to retrieve
        custom_config: Optional custom configuration (name, personality, backstory)
    
    Returns:
        PersonAgent instance
    """
    if custom_config:
        config = custom_config
    else:
        config = get_person_config(topic)
        if not config:
            raise ValueError(f"No predefined configuration found for topic: {topic}")
    
    return PersonAgent(
        name=config["name"],
        topic=topic,
        personality_type=config["personality"],
        backstory=config["backstory"],
        index=index,
        top_k=top_k,
    )