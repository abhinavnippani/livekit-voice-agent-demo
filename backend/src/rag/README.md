# RAG (Retrieval-Augmented Generation)

The RAG system provides functionality for loading PDF documents and querying them for context. It uses FAISS for vector storage and HuggingFace embeddings.

## Overview

The RAG system consists of:
- **PDF Loader**: Loads and chunks PDF documents
- **Vector Store**: FAISS-based vector database for semantic search
- **Retriever**: Queries the vector store to retrieve relevant context
- **RAG Service**: Main orchestrator that coordinates all components

## Adding PDF Documents

1. Place your PDF files in the `rag/data/` folder

2. Load a PDF file into a topic-specific collection:

```bash
uv run python -m rag.load_topic_pdf <topic> <pdf_filename>
# Example:
uv run python -m rag.load_topic_pdf interruption interruption_playbook.pdf
```

The script will automatically look for the PDF in the `rag/data/` folder. You can also provide an absolute path if the PDF is located elsewhere. Be sure to pass one of the configured topics (see `rag/person_agent.py`) so the document is routed to the correct FAISS collection.

This will:
- Load and parse the PDF
- Chunk the document into smaller pieces
- Generate embeddings and store them in a FAISS vector database
- Persist the vector store to disk (default: `./vector_db`)

## Testing the RAG Pipeline

To test the RAG pipeline and query the loaded documents:

```bash
uv run python -m rag.test_rag
```

This will:
- Initialize the RAG service
- Run sample queries against the loaded documents
- Display the retrieved context and document counts

## Using RAG in Your Code

You can also use the RAG service programmatically:

```python
from rag import get_rag_service

# Get the RAG service instance
rag = get_rag_service()

# Query the vector store
context = rag.query("What is the main topic?")
print(context)

# Get the number of documents in the collection
count = rag.get_collection_count()
print(f"Total chunks: {count}")
```

## Architecture

- **`pdf_loader.py`**: Handles PDF loading and text chunking
- **`vector_store.py`**: Manages FAISS vector store initialization and persistence
- **`retrieval.py`**: Handles querying and retrieving relevant context
- **`rag_service.py`**: Main service that orchestrates all components
- **`load_topic_pdf.py`**: CLI utility to load PDFs into topic-specific collections
- **`test_rag.py`**: Test script to verify the RAG pipeline

## Configuration

The RAG service uses default settings optimized for semantic search:
- **Chunk size**: 200 characters
- **Chunk overlap**: 20 characters
- **Top K**: 3 results per query
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector store**: FAISS with L2 distance

You can customize these settings when creating a `RAGService` instance directly.

