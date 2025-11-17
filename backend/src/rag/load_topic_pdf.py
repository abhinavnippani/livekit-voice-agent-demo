#!/usr/bin/env python3
"""
Utility script to load PDF files into topic-specific collections.
Usage: python -m rag.load_topic_pdf <topic> <pdf_filename>
"""
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.multi_agent_rag_service import get_multi_agent_rag_service
from rag.person_agent import PERSON_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_available_topics():
    """Print available topics."""
    print("\nAvailable topics:")
    for config in PERSON_CONFIGS:
        print(f"  - {config['topic']}: {config['name']} ({config['personality']})")


def main():
    """Load a PDF file into a topic-specific collection."""
    if len(sys.argv) < 3:
        print("Usage: python -m rag.load_topic_pdf <topic> <pdf_filename>")
        print("\nExample: python -m rag.load_topic_pdf interruption interruption_playbook.pdf")
        print_available_topics()
        print("\nNote: PDF files should be placed in the rag/data/ folder")
        sys.exit(1)
    
    topic = sys.argv[1]
    pdf_input = sys.argv[2]
    
    # Validate topic
    valid_topics = [config["topic"] for config in PERSON_CONFIGS]
    if topic not in valid_topics:
        print(f"Error: Invalid topic '{topic}'")
        print_available_topics()
        sys.exit(1)
    
    # Get the data folder path
    rag_dir = Path(__file__).parent
    data_dir = rag_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Determine the actual PDF path
    pdf_path_obj = Path(pdf_input)
    if pdf_path_obj.is_absolute():
        pdf_path = str(pdf_path_obj)
    else:
        pdf_path = str(data_dir / pdf_input)
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found: {pdf_path}")
        print(f"\nLooking for PDF in: {data_dir}")
        if data_dir.exists() and any(data_dir.iterdir()):
            print("Available files in data folder:")
            for file in data_dir.iterdir():
                if file.is_file() and file.suffix.lower() == '.pdf':
                    print(f"  - {file.name}")
        sys.exit(1)
    
    # Get person config for this topic
    person_config = next((c for c in PERSON_CONFIGS if c["topic"] == topic), None)
    
    logger.info("="*70)
    logger.info("LOADING PDF INTO TOPIC-SPECIFIC COLLECTION")
    logger.info("="*70)
    logger.info(f"Topic: {topic}")
    if person_config:
        logger.info(f"Expert: {person_config['name']}")
        logger.info(f"Personality: {person_config['personality']}")
    logger.info(f"PDF Path: {pdf_path}")
    logger.info("="*70)
    
    try:
        # Get multi-agent RAG service
        rag_service = get_multi_agent_rag_service()
        
        # Load PDF for specific topic
        logger.info(f"Loading PDF for topic: {topic}...")
        rag_service.load_pdf_for_topic(pdf_path, topic)
        
        # Get collection counts
        counts = rag_service.get_collection_counts()
        logger.info("\n" + "="*70)
        logger.info("✅ Successfully loaded PDF!")
        logger.info("="*70)
        logger.info("Collection counts:")
        for t, count in counts.items():
            status = "✓" if count > 0 else "○"
            logger.info(f"  {status} {t}: {count} chunks")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"❌ Error loading PDF: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()