# LiveKit Agent Backend

Python backend for the LiveKit voice agent.

## Setup

1. Install dependencies:
   ```bash
   uv sync
   # Or: pip install -e .
   ```

2. Create `.env` file:
   ```bash
   cp .env.example .env
   ```

3. Edit `.env` with your credentials:
   ```
   LIVEKIT_URL=wss://your-livekit-server.com
   LIVEKIT_API_KEY=your-api-key
   LIVEKIT_API_SECRET=your-api-secret
   ```
   
   **Note**: Only LiveKit API keys are needed! The agent uses LiveKit Inference to access AI models (STT, TTS, LLM), so you don't need separate API keys for OpenAI, AssemblyAI, or Cartesia.

4. Download required models (first time only):
   ```bash
   uv run livekit-agent download-files
   # Or: uv run python -m livekit_agent.agent download-files
   ```

## Running

### Development Mode
```bash
uv run livekit-agent dev
# Or: uv run python -m livekit_agent.agent dev
```

### Production Mode
```bash
uv run livekit-agent start
# Or: uv run python -m livekit_agent.agent start
```

## Token Server

To run the token server for frontend token generation:

1. Install token server dependencies:
   ```bash
   uv pip install -r requirements-token-server.txt
   ```

2. Run the server:
   ```bash
   uv run python token-server.py
   ```
   
   **Note**: Use `uv run` to ensure you're using the correct Python environment with all dependencies installed.

The server will run on `http://localhost:8080` and provide a `/api/token` endpoint.

