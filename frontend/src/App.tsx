import React, { useState, useEffect, useCallback, useRef } from 'react'
import {
  Room,
  RoomEvent,
  RemoteParticipant,
  Track,
  TrackPublication,
  RemoteTrackPublication,
  TranscriptionSegment,
  createLocalAudioTrack,
  LocalAudioTrack,
} from 'livekit-client'
import { generateToken } from './token-server'
import './App.css'

// Replace these with your LiveKit server details
const LIVEKIT_URL = import.meta.env.VITE_LIVEKIT_URL || 'wss://your-livekit-server.com'
const LIVEKIT_API_KEY = import.meta.env.VITE_LIVEKIT_API_KEY || ''
const LIVEKIT_API_SECRET = import.meta.env.VITE_LIVEKIT_API_SECRET || ''

function App() {
  const [room, setRoom] = useState<Room | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [transcription, setTranscription] = useState<string>('')
  const [isConnecting, setIsConnecting] = useState(false)
  const [localAudioTrack, setLocalAudioTrack] = useState<LocalAudioTrack | null>(null)
  
  // Use refs for cleanup to avoid stale closures
  const roomRef = useRef<Room | null>(null)
  const audioTrackRef = useRef<LocalAudioTrack | null>(null)
  
  // Track transcription segments by ID to handle streaming updates
  const transcriptionSegmentsRef = useRef<Map<string, { text: string; speaker: string; timestamp: number }>>(new Map())
  
  // Keep refs in sync with state
  useEffect(() => {
    roomRef.current = room
  }, [room])
  
  useEffect(() => {
    audioTrackRef.current = localAudioTrack
  }, [localAudioTrack])

  const connectToRoom = useCallback(async () => {
    if (isConnecting || isConnected) {
      console.log('Already connecting or connected, skipping...')
      return
    }

    setIsConnecting(true)
    
    // Prevent multiple simultaneous connection attempts
    let currentRoom: Room | null = null
    
    try {
      const roomName = `voice-agent-${Date.now()}`
      const participantName = 'user'  // Make sure this is unique and different from agent identity
      console.log('=== FRONTEND CONNECTION START ===')
      console.log('Generated room name:', roomName)
      console.log('Participant name:', participantName)

      // Generate token
      let token: string
      try {
        token = await generateToken(
          roomName,
          participantName,
          LIVEKIT_URL,
          LIVEKIT_API_KEY,
          LIVEKIT_API_SECRET
        )
      } catch (error) {
        console.error('Token generation failed:', error)
        alert('Token generation failed. Please check your configuration. See README for instructions.')
        setIsConnecting(false)
        return
      }

      // Create room instance - make sure it's not recreated on re-renders
      const newRoom = new Room({
        adaptiveStream: true,
        dynacast: true,
      })
      currentRoom = newRoom
      
      // Set room state IMMEDIATELY to prevent cleanup from disconnecting it
      setRoom(newRoom)

      // Set up transcription listener
      // TranscriptionReceived event provides an array of segments
      // Segments stream in with updates - we need to track by ID and update instead of append
      newRoom.on(RoomEvent.TranscriptionReceived, (
        segments: TranscriptionSegment[],
        participant?: RemoteParticipant,
        publication?: RemoteTrackPublication
      ) => {
        console.log('Transcription received:', segments.length, 'segment(s)')
        
        segments.forEach((segment) => {
          if (!segment.text || !segment.text.trim()) {
            return
          }
          
          // Get participant identity - check multiple sources
          let speaker = 'Unknown'
          
          // First, try the participant parameter (most reliable)
          if (participant) {
            speaker = participant.identity
          } 
          // Then try publication participant
          else if (publication?.participant) {
            speaker = publication.participant.identity
          }
          // If we have a track SID, try to find the participant by track
          else if (publication?.trackSid) {
            // Search all participants for this track
            for (const p of newRoom.remoteParticipants.values()) {
              if (p.trackPublications.has(publication.trackSid)) {
                speaker = p.identity
                break
              }
            }
            // If not found in remote, check local participant
            if (speaker === 'Unknown') {
              for (const pub of newRoom.localParticipant.trackPublications.values()) {
                if (pub.trackSid === publication.trackSid) {
                  speaker = newRoom.localParticipant.identity
                  break
                }
              }
            }
          }
          // Fallback: check if it's from local participant (user) or agent
          else {
            // Check if any agent participant has tracks
            const agentParticipant = Array.from(newRoom.remoteParticipants.values()).find(p => p.isAgent)
            if (agentParticipant && agentParticipant.trackPublications.size > 0) {
              // If we can't determine, assume it's from the agent if agent exists
              // Otherwise assume it's from the user
              speaker = agentParticipant.identity
            } else {
              // User transcription - use local participant identity
              speaker = newRoom.localParticipant?.identity || 'user'
            }
          }
          
          console.log('Transcription segment - speaker:', speaker, 'text:', segment.text.substring(0, 50) + '...')
          
          // Use startTime (in milliseconds) for timestamp, fallback to firstReceivedTime
          const timestampMs = segment.startTime || segment.firstReceivedTime || Date.now()
          const timestamp = new Date(timestampMs).toLocaleTimeString()
          
          // Update or add segment to our tracking map
          // If segment ID already exists, it will be updated (for streaming)
          transcriptionSegmentsRef.current.set(segment.id, {
            text: segment.text,
            speaker: speaker,
            timestamp: timestampMs
          })
          
          // Rebuild transcription from all segments
          // Each unique segment ID represents one distinct message/utterance
          // Sort by timestamp to maintain chronological order
          const allSegments = Array.from(transcriptionSegmentsRef.current.entries())
            .map(([id, seg]) => ({ id, ...seg }))
            .sort((a, b) => a.timestamp - b.timestamp)
          
          // Group segments by their ID - each ID = one message
          // When same ID appears multiple times, it's a streaming update (we keep the latest)
          const messagesBySegmentId = new Map<string, { speaker: string; text: string; startTime: number }>()
          
          allSegments.forEach((seg) => {
            // Each segment ID represents a distinct message
            // If we've seen this ID before, update it (streaming update)
            // Otherwise, create a new message
            if (!messagesBySegmentId.has(seg.id)) {
              messagesBySegmentId.set(seg.id, {
                speaker: seg.speaker,
                text: seg.text.trim(),
                startTime: seg.timestamp
              })
            } else {
              // Update existing message (streaming update)
              const existing = messagesBySegmentId.get(seg.id)!
              existing.text = seg.text.trim()
              // Keep the earliest timestamp for this message
              if (seg.timestamp < existing.startTime) {
                existing.startTime = seg.timestamp
              }
            }
          })
          
          // Convert to array and sort by timestamp
          const messages = Array.from(messagesBySegmentId.values())
            .sort((a, b) => a.startTime - b.startTime)
          
          // Format each message as a line - each message gets its own line
          const lines = messages.map(msg => {
            const formattedTime = new Date(msg.startTime).toLocaleTimeString()
            return `[${formattedTime}] ${msg.speaker}: ${msg.text.trim()}`
          })
          
          // Update transcription state - each line is a separate message
          setTranscription(lines.join('\n'))
          
          // Clean up finalized segments (optional - keep them for history)
          // if (segment.final) {
          //   // Segment is finalized, could clean up old segments if needed
          // }
        })
      })

      // Set up connection events
      newRoom.on(RoomEvent.Connected, async () => {
        console.log('Connected to room')
        
        // Request microphone access and publish audio track
        try {
          const audioTrack = await createLocalAudioTrack({
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          })
          await newRoom.localParticipant.publishTrack(audioTrack)
          setLocalAudioTrack(audioTrack)
          console.log('Audio track published')
        } catch (error) {
          console.error('Failed to publish audio track:', error)
          alert('Failed to access microphone. Please check your permissions.')
          setIsConnecting(false)
          return
        }
        
        // Wait briefly for agent to connect (optional but improves UX)
        // The agent should connect within a few seconds, but we don't want to block too long
        const waitForAgent = new Promise<void>((resolve) => {
          const timeout = setTimeout(() => {
            console.log('Continuing without waiting for agent (agent may connect later)...')
            resolve()
          }, 3000)  // Reduced to 3 seconds to avoid deadlock
          
          const checkForAgent = () => {
            // Check existing participants
            for (const participant of newRoom.remoteParticipants.values()) {
              if (participant.identity.includes('agent') || participant.isAgent) {
                console.log('Agent participant found:', participant.identity)
                clearTimeout(timeout)
                resolve()
                return
              }
            }
          }
          
          // Check immediately
          checkForAgent()
          
          // Also listen for new participants
          newRoom.on(RoomEvent.ParticipantConnected, (participant) => {
            if (participant.identity.includes('agent') || participant.isAgent) {
              console.log('Agent participant connected:', participant.identity)
              clearTimeout(timeout)
              resolve()
            }
          })
        })
        
        await waitForAgent
        
        // Don't disconnect if agent isn't found - just continue
        // The agent may connect later and the session will attach
        console.log('Setting connected state - agent may connect later')
        setIsConnected(true)
        setIsConnecting(false)
        
        // Log current room state
        console.log('Current room state:')
        console.log('  - Remote participants:', newRoom.remoteParticipants.size)
        newRoom.remoteParticipants.forEach((participant) => {
          console.log('  - Participant:', participant.identity, 'isAgent:', participant.isAgent)
        })
      })

      newRoom.on(RoomEvent.Disconnected, (reason) => {
        console.log('=== DISCONNECTED FROM ROOM ===')
        console.log('Disconnect reason:', reason)
        console.log('Disconnect reason code:', reason?.code)
        console.log('Disconnect reason message:', reason?.message)
        setIsConnected(false)
        setTranscription('')
        setIsConnecting(false)
        // Clear transcription segments on disconnect
        transcriptionSegmentsRef.current.clear()
      })

      newRoom.on(RoomEvent.ConnectionQualityChanged, (quality, participant) => {
        console.log('Connection quality changed:', quality, 'for participant:', participant?.identity)
      })

      newRoom.on(RoomEvent.Reconnecting, () => {
        console.log('Reconnecting to room...')
      })

      newRoom.on(RoomEvent.Reconnected, () => {
        console.log('Reconnected to room')
      })

      newRoom.on(RoomEvent.MediaDevicesError, (error) => {
        console.error('Media devices error:', error)
      })

      newRoom.on(RoomEvent.ParticipantConnected, (participant) => {
        console.log('=== PARTICIPANT CONNECTED EVENT (Frontend) ===')
        console.log('Participant identity:', participant.identity)
        console.log('Participant SID:', participant.sid)
        console.log('Is agent:', participant.isAgent)
        console.log('Track publications:', participant.trackPublications.size)
      })

      newRoom.on(RoomEvent.TrackSubscribed, (
        track: Track,
        publication: RemoteTrackPublication,
        participant: RemoteParticipant
      ) => {
        console.log('=== TRACK SUBSCRIBED EVENT ===')
        console.log('Track kind:', track.kind)
        console.log('From participant:', participant.identity)
        console.log('Is agent:', participant.isAgent)
        console.log('Track SID:', track.sid)
        
        if (track.kind === Track.Kind.Audio) {
          console.log('Audio track subscribed - setting up playback')
          // Create audio element and play the track
          // Use a unique ID to avoid creating multiple elements for the same track
          const audioElementId = `audio-${participant.identity}-${track.sid}`
          let audioElement = document.getElementById(audioElementId) as HTMLAudioElement
          
          if (!audioElement) {
            audioElement = document.createElement('audio')
            audioElement.id = audioElementId
            audioElement.autoplay = true
            audioElement.playsInline = true
            document.body.appendChild(audioElement)
            console.log('Created audio element for track:', track.sid)
          }
          
          track.attach(audioElement)
          console.log('Audio track attached to element, should start playing')
          
          // Log when audio starts playing
          audioElement.onplay = () => {
            console.log('Audio started playing from participant:', participant.identity)
          }
          
          audioElement.onerror = (error) => {
            console.error('Audio playback error:', error)
          }
        }
      })

      // Connect to room
      console.log('Connecting to room:', roomName)
      console.log('LiveKit URL:', LIVEKIT_URL)
      
      try {
        await newRoom.connect(LIVEKIT_URL, token)
        console.log('Room connected successfully')
        console.log('Room name:', newRoom.name)
        console.log('Room SID:', newRoom.sid)
        console.log('Local participant identity:', newRoom.localParticipant.identity)
        console.log('Connection state:', newRoom.connectionState)
        
        // Room is already set above, but verify it's still the same instance
        if (currentRoom !== newRoom) {
          console.warn('Room instance changed during connection!')
        }
      } catch (connectError) {
        console.error('Failed to connect to room:', connectError)
        // Clean up on connection failure
        if (currentRoom) {
          currentRoom.disconnect().catch(console.error)
          setRoom(null)
        }
        alert('Failed to connect to room. Check your configuration.')
        setIsConnecting(false)
        return
      }
      
      // Monitor connection state
      newRoom.on(RoomEvent.ConnectionStateChanged, (state) => {
        console.log('Connection state changed:', state)
        if (state === 'disconnected') {
          console.warn('Room disconnected unexpectedly!')
        }
      })
    } catch (error) {
      console.error('Failed to connect:', error)
      // Clean up on error
      if (currentRoom) {
        currentRoom.disconnect().catch(console.error)
        setRoom(null)
      }
      alert('Failed to connect to room. Check your configuration.')
      setIsConnecting(false)
    }
    // Remove dependencies that cause re-renders - only depend on what's truly needed
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const disconnectFromRoom = useCallback(async () => {
    console.log('=== MANUAL DISCONNECT REQUESTED ===')
    if (localAudioTrack) {
      console.log('Stopping local audio track')
      localAudioTrack.stop()
      localAudioTrack.detach()
      setLocalAudioTrack(null)
    }
    if (room) {
      console.log('Disconnecting room:', room.name)
      try {
        await room.disconnect()
        console.log('Room disconnected successfully')
      } catch (error) {
        console.error('Error disconnecting room:', error)
      }
      setRoom(null)
      setIsConnected(false)
      setTranscription('')
      // Clear transcription segments on disconnect
      transcriptionSegmentsRef.current.clear()
    }
  }, [room, localAudioTrack])

  // Cleanup on unmount only - NOT on every room/track change
  // This prevents accidental disconnects during re-renders
  useEffect(() => {
    return () => {
      // Only cleanup on component unmount
      // Use refs to get current values, not stale closures
      console.log('Component unmounting, cleaning up...')
      const currentTrack = audioTrackRef.current
      const currentRoom = roomRef.current
      
      if (currentTrack) {
        console.log('Cleaning up audio track on unmount')
        currentTrack.stop()
        currentTrack.detach()
      }
      if (currentRoom) {
        console.log('Disconnecting room on unmount:', currentRoom.name)
        currentRoom.disconnect().catch(console.error)
      }
    }
    // Empty dependency array = only run on unmount
  }, [])

  return (
    <div className="app">
      <div className="container">
        <h1>MockEvents</h1>
        <h2 className="subheading">Voice Agents Mixer: Streaming, Latency, Interrupt</h2>
        
        <div className="status">
          <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`} />
          <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
        </div>

        <div className="transcription-box">
          <h2>Live Transcription</h2>
          <div className="transcription-text">
            {transcription || 'Transcription will appear here...'}
          </div>
        </div>

        <div className="controls">
          {!isConnected ? (
            <button
              onClick={connectToRoom}
              disabled={isConnecting}
              className="btn btn-start"
            >
              {isConnecting ? 'Connecting...' : 'Start'}
            </button>
          ) : (
            <button
              onClick={disconnectFromRoom}
              className="btn btn-stop"
            >
              Stop
            </button>
          )}
        </div>

        <div className="info">
          <p>Click Start to begin interacting with the voice agent.</p>
          <p>Your conversation will be transcribed in real-time.</p>
        </div>
      </div>
    </div>
  )
}

export default App

