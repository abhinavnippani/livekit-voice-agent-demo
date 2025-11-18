# MockEvents: Voice AI Networking Mixer

A voice-first networking event simulation powered by LiveKit Inference and multi-agent RAG. Talk with three topic experts (interruptions, streaming, latency) who can hand off conversations based on your questions.

## Architecture

**Flow:** Frontend (React) → LiveKit Server → Voice Agent → Multi-Agent RAG → Topic Expert Response

- **Frontend:** React + TypeScript + Vite, generates LiveKit tokens, streams audio/transcriptions
- **Backend:** LiveKit Agent with Inference (AssemblyAI STT, GPT-4o-mini LLM, Cartesia Sonic TTS, Silero VAD)
- **RAG:** LlamaIndex + FAISS with topic-specific collections, HuggingFace `all-MiniLM-L6-v2` embeddings
- **Orchestration:** Routes queries to topic experts, handles handoffs, tracks conversation history

## RAG Details

- **Corpus:** 3 PDFs (one per topic) from `backend/src/rag/data`
- **Storage:** Local FAISS with `SingleFAISSMultiCollection`, persisted in `backend/vector_db`
- **Chunking:** 200-char chunks, 20-char overlap, topic-tagged metadata
- **Retrieval:** Topic-filtered `VectorStoreIndex` per agent, top-k=3
- **Routing:** Keyword-based topic detection with explicit handoff support

## Setup

**Prerequisites:** Python 3.11+, `uv`, Node.js 18+, LiveKit API keys with Inference enabled

1. **Backend:** `cd backend && uv sync && cp .env.example .env` → set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` → `uv run livekit-agent download-files`
2. **Frontend:** `cd frontend && npm install && cp .env.example .env` → set `VITE_LIVEKIT_*`
3. **Token server (prod):** `cd backend && uv pip install -r requirements-token-server.txt && uv run python token-server.py`
4. **Run:** `uv run livekit-agent dev` (backend) + `npm run dev` (frontend) → open `http://localhost:3000`

## Design Notes

- Single `query_networking_event` tool returns persona + context in one call
- Context-aware handoff messages: LLM generates personalized transitions between experts
- Keyword-based topic routing with explicit handoff support (semantic router planned)
- Conversation history tracking for context-aware responses
- Local FAISS (suitable for demos, not multi-user scale)
- Preemptive generation + interruption enabled for natural conversation
- Adding topics requires PDF + persona config in `person_agent.py`

Happy Networking!

