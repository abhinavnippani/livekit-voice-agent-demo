# Welcome to MockEvents

We build lightweight simulations of everyday events. Today’s demo drops you into a networking mixer.

Today's network is about Voice AI, where we are focusing on 3 topics - interruptions, Streaming and Latency. There are 3 other experts for these topics in the event. Feel free to network with them.

Here’s a concise technical brief on the LiveKit Voice Agent that powers this mixer:

# LiveKit Voice Agent

A compact voice-first demo that blends LiveKit Inference (STT, LLM, TTS) with a multi-agent RAG back end and a React control surface.

## End-to-End Flow

1. **Frontend (Vite + React + `livekit-client`)** acquires microphone access, generates a LiveKit token via the bundled token helper (or optional Flask token service), then joins a unique room and streams audio + UI telemetry.
2. **LiveKit Cloud / Self-hosted LiveKit Server** bridges the user stream with the hosted LiveKit Voice Agent runtime.
3. **Backend agent (`livekit_agent`)** boots with LiveKit Inference defaults (AssemblyAI STT, GPT-4o-mini LLM, Cartesia Sonic TTS, Silero VAD). It exposes a single tool, `query_networking_event`, which calls the multi-agent RAG service.
4. **Multi-agent RAG** routes the user query to topic experts (interruption, latency, streaming). Each agent retrieves topic-scoped context from its FAISS collection, crafts a persona-aware prompt, and hands the result back through LiveKit for synthesis and playback.
5. **Frontend** renders live transcription, remote audio, and connection state.

## RAG Integration Snapshot

- **Corpus:** Three PDFs (interruption, latency, streaming) ingested from `backend/src/rag/data`, one per persona/topic.
- **Vector DB:** Local FAISS index managed by `SingleFAISSMultiCollection`; persisted under `backend/vector_db`.
- **Embeddings & Frameworks:** LlamaIndex + HuggingFace `all-MiniLM-L6-v2`.
- **Chunking:** PDF loader splits docs into ~200-char chunks with 20-char overlap; metadata tags each chunk with the agent topic.
- **Retrieval:** Topic-specific `VectorStoreIndex` per agent; `Retriever` enforces metadata filters so each persona only accesses its own context.
- **Orchestration:** `Orchestrator` tracks conversation history, detects topics/explicit handoffs, rotates personas, and produces system prompts that embed peer awareness plus summaries.

## Tooling & Frameworks

- **Voice + Realtime:** LiveKit Inference (STT, LLM, TTS), LiveKit Agent SDK, Silero VAD.
- **Knowledge:** LlamaIndex, HuggingFace embeddings, FAISS.
- **Web:** React 18, TypeScript, Vite, `livekit-client`.
- **Ops:** `uv` for Python envs, npm for frontend, optional Flask token server.
- **AI Tools:** Cursor for rapid code navigation, refactors, web search and docs edits.

## Setup (Local or Cloud)

1. **Backend prerequisites:** Python 3.11+, `uv`, LiveKit API key/secret with Inference enabled.
2. **Frontend prerequisites:** Node.js 18+.
3. **Backend install:** `cd backend && uv sync && cp .env.example .env` → set `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` → `uv run livekit-agent download-files`.
4. **Frontend install:** `cd frontend && npm install && cp .env.example .env` → set `VITE_LIVEKIT_*`.
5. **Token generation:** For dev you can keep secrets in `.env` and call `generateToken`. For production run `cd backend && uv pip install -r requirements-token-server.txt && uv run python token-server.py` and point the frontend to `/api/token`.
6. **Run dev stack:** `uv run livekit-agent dev` (backend) + `npm run dev` (frontend) + token server. Open `http://localhost:3000`, click **Start**, allow the mic; backend logs confirm expert selection and RAG hits.
7. **Web deploy:** `npm run build` for static hosting, deploy backend agent (or LiveKit Cloud agent hosting) plus HTTPS token server; update envs accordingly.

## Design Decisions & Assumptions

- **Voice agent design:** A single LiveKit Agent hosts multiple personas via one `query_networking_event` tool. That tool returns the active expert’s personality prompt plus freshly retrieved context, so the LLM only needs a single tool call per turn for both knowledge and persona.
- **Trade-offs & limitations:**  
  - Topic detection uses keyword heuristics; no semantic router yet.  
  - FAISS runs in-process with local disk persistence—good for demos, not multi-user scale.  
  - Token helper stores secrets in the browser if you skip the token server (dev-only).  
  - Only three personas/topics are bundled; adding more requires PDFs + persona config.
- **Hosting assumptions:** LiveKit server (cloud or self-hosted: localhost, AWS, GCP, etc.) is reachable over HTTPS/WSS; backend can reach the same server with low latency; token server runs behind HTTPS in production. Frontend expects the Vite dev server locally or any static host with ENV injection.
- **RAG assumptions:** PDFs live under `backend/src/rag/data`, chunking stays at 200/20 for latency reasons, FAISS stores per-topic collections, embeddings remain `all-MiniLM-L6-v2`. Changing chunk size or embeddings requires re-ingesting documents.
- **LiveKit agent specifics:**  
  - STT: `assemblyai/universal-streaming`.  
  - LLM: `openai/gpt-4o-mini`.  
  - TTS: `cartesia/sonic-3`.  
  - VAD: Silero with aggressive settings for natural turn-taking.  
  - Session opts: preemptive generation + interruption enabled to keep the mixer conversational.
- **Future improvements:** Replace keyword routing with embedding classifiers, move FAISS to a managed vector DB, add evals/tests for retrieval coverage, and productionize the token service with auth + caching.

Happy Networking!

