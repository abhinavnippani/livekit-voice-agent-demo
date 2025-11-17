"""
Quick sanity check for the multi-agent networking event setup.

Usage:
    uv run python -m rag.test_rag

Assumes you have already:
    1. Deleted the previous vector DB (`rm -rf vector_db`)
    2. Loaded the three PDFs:
       uv run python -m rag.load_topic_pdf interruption rag/data/interruption.pdf
       uv run python -m rag.load_topic_pdf latency rag/data/latency.pdf
       uv run python -m rag.load_topic_pdf streaming rag/data/streaming.pdf
"""

from rag.multi_agent_rag_service import get_multi_agent_rag_service


def pretty_print(title: str, content: str):
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(content.strip() if isinstance(content, str) else content)
    print()


def run_queries():
    service = get_multi_agent_rag_service(
        topics=["interruption", "latency", "streaming"]
    )

    pretty_print("Collection Counts", service.get_collection_counts())

    queries = [
        ("interruption", "How should a voice agent handle user interruptions gracefully?"),
        ("latency", "What are the biggest causes of latency in a live voice pipeline?"),
        ("streaming", "How do you keep streaming smooth when bandwidth drops suddenly?"),
    ]

    for topic, query in queries:
        target_agent = service.orchestrator.topic_to_agent.get(topic)
        if target_agent:
            service.orchestrator.set_current_person(target_agent)

        result = service.query(query)
        person = result["person"]
        context_preview = (result.get("context") or "").strip()[:500]
        pretty_print(
            f"Response routed to {person} ({topic})\nQuery: {query}",
            f"Context preview:\n{context_preview or '[no context retrieved]'}",
        )


if __name__ == "__main__":
    run_queries()