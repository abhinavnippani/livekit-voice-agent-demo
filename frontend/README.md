# LiveKit Voice Agent Frontend

React frontend for the LiveKit voice agent.

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create `.env` file:
   ```bash
   cp .env.example .env
   ```

3. Edit `.env` with your credentials:
   ```
   VITE_LIVEKIT_URL=wss://your-livekit-server.com
   VITE_LIVEKIT_API_KEY=your-api-key
   VITE_LIVEKIT_API_SECRET=your-api-secret
   ```

## Running

### Development
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Production Build
```bash
npm run build
```

## Features

- Start/Stop voice interaction
- Live transcription display
- Real-time audio streaming
- Connection status indicator

## Token Generation

The frontend needs tokens to connect to LiveKit. You have two options:

1. **Use the token server** (recommended for development):
   - Run the token server from the backend directory
   - The frontend will automatically call `/api/token`

2. **Generate tokens client-side** (development only):
   - Set `VITE_LIVEKIT_API_SECRET` in `.env`
   - Note: This is NOT secure for production

For production, always generate tokens on your backend server.

