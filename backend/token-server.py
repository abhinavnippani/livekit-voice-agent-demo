#!/usr/bin/env python3
"""
Simple token server for LiveKit
Run this server to generate tokens for your frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from livekit.api import AccessToken, VideoGrants
from dotenv import load_dotenv
from pathlib import Path
import os

# Load .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

app = Flask(__name__)
CORS(app)

# Get LiveKit credentials from environment
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")


@app.route("/api/token", methods=["POST"])
def generate_token():
    """Generate a LiveKit access token"""
    try:
        data = request.get_json()
        room_name = data.get("roomName", "default-room")
        participant_name = data.get("participantName", "user")

        if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
            return jsonify({"error": "LiveKit credentials not configured"}), 500

        # Create access token
        token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
            .with_identity(participant_name) \
            .with_name(participant_name) \
            .with_grants(VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            ))

        return jsonify({
            "token": token.to_jwt(),
            "url": LIVEKIT_URL,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("Starting token server on http://localhost:8080")
    print("Make sure to set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET")
    app.run(host="0.0.0.0", port=8080, debug=True)

