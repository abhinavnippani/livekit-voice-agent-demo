#!/usr/bin/env python3
"""
LiveKit Voice Agent
A simple voice AI assistant using LiveKit Agents with LiveKit Inference
Uses only LiveKit API keys - no external provider keys needed
"""

import asyncio
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    AutoSubscribe,
)
from livekit.agents.voice import AgentSession
from livekit import rtc

from .agent_config import create_agent, get_greeting_message

# Load .env file from the backend directory
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def entrypoint(ctx: JobContext):
    """Entrypoint for the LiveKit agent job"""
    logger.info("=== AGENT ENTRYPOINT ===")
    logger.info("Agent starting for room: %s", ctx.room.name)
    logger.info("Job ID: %s", ctx.job.id if hasattr(ctx.job, 'id') else 'N/A')
    logger.info("Room name from context: %s", ctx.room.name)
    
    # Event to signal when a participant with SUBSCRIBED audio track is ready
    participant_ready_event = asyncio.Event()
    participant_with_audio = None
    
    def has_subscribed_audio_track(participant: rtc.RemoteParticipant) -> bool:
        """Check if participant has a subscribed audio track with actual track data"""
        for pub in participant.track_publications.values():
            if pub.kind == rtc.TrackKind.KIND_AUDIO:
                # Track must be both subscribed AND have the actual track object
                if pub.subscribed and pub.track is not None:
                    return True
        return False
    
    def check_and_signal_participant(participant: rtc.RemoteParticipant):
        """Check if participant has subscribed audio track and signal if ready"""
        nonlocal participant_with_audio
        if has_subscribed_audio_track(participant):
            logger.info("Participant %s has subscribed audio track, ready for session", participant.identity)
            participant_with_audio = participant
            participant_ready_event.set()
        else:
            # Check what's missing
            audio_pubs = [p for p in participant.track_publications.values() if p.kind == rtc.TrackKind.KIND_AUDIO]
            if audio_pubs:
                for pub in audio_pubs:
                    logger.debug("Participant %s audio track %s: subscribed=%s, track=%s", 
                               participant.identity, pub.sid, pub.subscribed, "present" if pub.track else "none")
            else:
                logger.info("Participant %s connected but no audio track yet", participant.identity)
    
    # Set up participant event handlers BEFORE connecting
    # This ensures we catch participants that connect during or after connection
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info("=== PARTICIPANT CONNECTED EVENT ===")
        logger.info("Participant identity: %s", participant.identity)
        logger.info("Participant SID: %s", participant.sid if hasattr(participant, 'sid') else 'N/A')
        logger.info("Participant kind: %s", participant.kind if hasattr(participant, 'kind') else 'N/A')
        logger.info("Number of track publications: %d", len(participant.track_publications))
        # Log tracks when participant connects
        for track_sid, publication in participant.track_publications.items():
            if publication.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info("Participant %s has audio track: %s (subscribed=%s)", 
                           participant.identity, track_sid, publication.subscribed)
        # Check if this participant is ready
        check_and_signal_participant(participant)
    
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info("Participant disconnected: %s", participant.identity)
        nonlocal participant_with_audio
        if participant_with_audio and participant_with_audio.identity == participant.identity:
            logger.warning("Participant %s with audio track disconnected", participant.identity)
            participant_with_audio = None
            # Only clear the event if session hasn't started yet
            # (We'll check this by seeing if session exists and is started)
            # For now, we'll keep the event set to prevent re-waiting if session is about to start
    
    def on_track_published(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        if publication.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info("Audio track published by participant: %s, track: %s", participant.identity, publication.sid)
            logger.info("Track subscription status: subscribed=%s", publication.subscribed)
            # AutoSubscribe.AUDIO_ONLY should handle subscription automatically
            # Just log and wait for track_subscribed event
    
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        """Track subscribed event - this is when we actually have the track data"""
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info("Audio track subscribed for participant: %s, track: %s", participant.identity, track.sid)
            # Now check if participant is ready (has subscribed track)
            check_and_signal_participant(participant)
    
    ctx.room.on("participant_connected", on_participant_connected)
    ctx.room.on("participant_disconnected", on_participant_disconnected)
    ctx.room.on("track_published", on_track_published)
    ctx.room.on("track_subscribed", on_track_subscribed)
    
    # Connect to the room
    logger.info("Connecting to room: %s", ctx.room.name)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info("Agent connected to room: %s", ctx.room.name)
    logger.info("Room connection state: %s", ctx.room.connection_state)
    # Room.sid might be a property or coroutine - skip for now to avoid errors
    # logger.info("Room SID: %s", ctx.room.sid)  # Skip SID logging to avoid coroutine issues
    
    # Log local participant info
    if ctx.room.local_participant:
        logger.info("Local participant (agent): %s", ctx.room.local_participant.identity)
    
    # Check existing participants after connection
    # IMPORTANT: Participants that connected BEFORE we set up event handlers
    # won't trigger events, so we need to manually process them
    logger.info("Checking for existing participants...")
    logger.info("Remote participants count: %d", len(ctx.room.remote_participants))
    
    # Manually trigger handlers for existing participants
    # This handles the case where participant connected before event handlers were set up
    for identity, participant in ctx.room.remote_participants.items():
        logger.info("=== PROCESSING EXISTING PARTICIPANT (connected before handlers) ===")
        logger.info("Participant identity: %s", identity)
        logger.info("Participant SID: %s", participant.sid if hasattr(participant, 'sid') else 'N/A')
        logger.info("Number of track publications: %d", len(participant.track_publications))
        # Manually call the handler
        on_participant_connected(participant)
        # Also check all their tracks
        for track_sid, publication in participant.track_publications.items():
            if publication.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info("Processing existing audio track: %s", track_sid)
                on_track_published(publication, participant)
                # If already subscribed, also trigger track_subscribed
                if publication.subscribed and publication.track:
                    logger.info("Track already subscribed, triggering track_subscribed handler")
                    on_track_subscribed(publication.track, publication, participant)
    
    # Wait briefly for a participant with SUBSCRIBED audio track
    # Use a short timeout to avoid deadlock with frontend waiting for agent
    # AgentSession can attach to participants as they join after session starts
    logger.info("=== WAITING FOR PARTICIPANT ===")
    logger.info("Participant ready event set: %s", participant_ready_event.is_set())
    logger.info("Current remote participants: %d", len(ctx.room.remote_participants))
    
    # Also set up a periodic check for participants that might join
    async def periodic_participant_check():
        """Periodically check for new participants and process them"""
        while not participant_ready_event.is_set():
            await asyncio.sleep(0.5)
            current_count = len(ctx.room.remote_participants)
            if current_count > 0:
                logger.info("Periodic check: Found %d participant(s)", current_count)
                for identity, participant in ctx.room.remote_participants.items():
                    # Process if we haven't seen this participant yet
                    if not participant_ready_event.is_set():
                        logger.info("Processing participant from periodic check: %s", identity)
                        on_participant_connected(participant)
                        # Check all tracks
                        for track_sid, publication in participant.track_publications.items():
                            if publication.kind == rtc.TrackKind.KIND_AUDIO:
                                on_track_published(publication, participant)
                                if publication.subscribed and publication.track:
                                    on_track_subscribed(publication.track, publication, participant)
    
    if not participant_ready_event.is_set():
        logger.info("No participant with subscribed audio track found, waiting briefly...")
        # Start periodic checking
        check_task = asyncio.create_task(periodic_participant_check())
        try:
            # Wait up to 5 seconds - short enough to avoid deadlock, long enough for fast connections
            await asyncio.wait_for(participant_ready_event.wait(), timeout=5.0)
            check_task.cancel()
            logger.info("Participant with subscribed audio track is ready: %s", 
                       participant_with_audio.identity if participant_with_audio else "unknown")
        except asyncio.TimeoutError:
            check_task.cancel()
            logger.info("No participant yet, but starting session anyway. Session will attach when participant joins.")
            # Final check of existing participants
            if len(ctx.room.remote_participants) > 0:
                logger.info("Found participants but tracks may not be subscribed yet.")
                logger.info("AutoSubscribe should handle subscription automatically, waiting for track_subscribed event...")
                # Give AutoSubscribe a moment to work
                await asyncio.sleep(1.0)
                for identity, participant in ctx.room.remote_participants.items():
                    check_and_signal_participant(participant)
    else:
        logger.info("Participant with subscribed audio track already ready: %s", 
                   participant_with_audio.identity if participant_with_audio else "unknown")
    
    # Create agent using configuration from agent_config
    agent = create_agent()
    
    # Create agent session
    session = AgentSession()
    
    # Create an event to wait for room disconnection
    disconnect_event = asyncio.Event()
    
    # Set up room connection state change handler
    def on_connection_state_changed(connection_state):
        if connection_state == rtc.ConnectionState.CONN_DISCONNECTED:
            logger.info("Room disconnected")
            disconnect_event.set()
    
    ctx.room.on("connection_state_changed", on_connection_state_changed)
    
    # Final verification: ensure participant still has subscribed audio track
    logger.info("Final verification before session start...")
    logger.info("  - Remote participants: %d", len(ctx.room.remote_participants))
    
    # Verify we have a participant with subscribed audio track
    ready_participant = None
    for identity, participant in ctx.room.remote_participants.items():
        audio_tracks = []
        subscribed_tracks = []
        for pub in participant.track_publications.values():
            if pub.kind == rtc.TrackKind.KIND_AUDIO:
                audio_tracks.append(pub)
                if pub.subscribed and pub.track is not None:
                    subscribed_tracks.append(pub)
                elif not pub.subscribed:
                    logger.info("Audio track %s for participant %s not subscribed yet (AutoSubscribe should handle this)", 
                               pub.sid, identity)
        
        logger.info("  - Participant %s: %d audio track(s), %d subscribed with track data", 
                   identity, len(audio_tracks), len(subscribed_tracks))
        
        if subscribed_tracks:
            ready_participant = participant
            logger.info("    Ready tracks: %s", [t.sid for t in subscribed_tracks])
    
    if ready_participant is None:
        logger.info("No participant with subscribed audio track found yet, but starting session anyway...")
        logger.info("Session will attach to participants as they join and publish audio tracks")
    else:
        logger.info("Participant %s ready with %d subscribed audio track(s), starting session", 
                   ready_participant.identity, len([p for p in ready_participant.track_publications.values() 
                                                    if p.kind == rtc.TrackKind.KIND_AUDIO and p.subscribed and p.track]))
    
    # Start the agent session with the agent and room
    # Note: When using an STT model (like AssemblyAI), AgentSession automatically:
    # - Processes audio through the STT model
    # - Publishes transcriptions to the room
    # - Frontend receives them via TranscriptionReceived events
    # The Agent class automatically handles the conversation flow:
    # - User speaks -> STT transcribes -> LLM processes -> TTS speaks response
    logger.info("Starting agent session...")
    logger.info("Agent configured with STT: %s", agent.stt if hasattr(agent, 'stt') else 'N/A')
    logger.info("Agent configured with LLM: %s", agent.llm if hasattr(agent, 'llm') else 'N/A')
    logger.info("Agent configured with TTS: %s", agent.tts if hasattr(agent, 'tts') else 'N/A')
    logger.info("Agent configured with VAD: %s", agent.vad if hasattr(agent, 'vad') else 'N/A')
    try:
        await session.start(agent, room=ctx.room)
        logger.info("Agent session started successfully")
        logger.info("Transcriptions from STT will be automatically published to the room")
    except Exception as e:
        logger.error("Failed to start agent session: %s", e, exc_info=True)
        raise
    
    # Log session state after start
    if hasattr(session, 'input') and session.input:
        logger.info("Session input configured: %s", type(session.input).__name__)
    if hasattr(session, 'output') and session.output:
        logger.info("Session output configured: %s", type(session.output).__name__)
    
    # Send initial greeting message from agent config
    # This ensures the agent starts the conversation and verifies TTS output
    try:
        greeting_message = get_greeting_message()
        logger.info("Sending initial greeting message: %s", greeting_message)
        await session.say(greeting_message, allow_interruptions=True)
        logger.info("Greeting message sent successfully")
    except Exception as e:
        logger.error("Failed to send greeting message: %s", e, exc_info=True)
        # Don't raise - continue even if greeting fails, as the agent should still work
    
    # Wait for the room to disconnect
    await disconnect_event.wait()
    logger.info("Agent session ending")


def main():
    """Main entry point for the agent"""
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )


if __name__ == "__main__":
    main()

