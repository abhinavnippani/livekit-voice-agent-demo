import React, { useState, useEffect, useCallback, useRef } from 'react'
import {
  Room,
  RoomEvent,
  RemoteParticipant,
  Participant,
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

// Message type for chat interface
interface Message {
  id: string
  speaker: string
  text: string
  timestamp: number
  isUser: boolean
}

// Message component for individual chat bubbles
const MessageBubble: React.FC<{ message: Message }> = ({ message }) => {
  const formattedTime = new Date(message.timestamp).toLocaleTimeString([], { 
    hour: '2-digit', 
    minute: '2-digit' 
  })
  
  return (
    <div className={`message ${message.isUser ? 'message-user' : 'message-agent'}`}>
      <div className="message-content">
        <div className="message-header">
          <span className="message-speaker">{message.speaker}</span>
          <span className="message-time">{formattedTime}</span>
        </div>
        <div className="message-text">{message.text}</div>
      </div>
    </div>
  )
}

function App() {
  const [room, setRoom] = useState<Room | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [messages, setMessages] = useState<Message[]>([])
  const [isConnecting, setIsConnecting] = useState(false)
  const [localAudioTrack, setLocalAudioTrack] = useState<LocalAudioTrack | null>(null)
  const [hasAgentSpoken, setHasAgentSpoken] = useState(false)
  const [hasStarted, setHasStarted] = useState(false)
  const [currentSpeaker, setCurrentSpeaker] = useState<string | null>(null)
  const [userName, setUserName] = useState<string>('')
  const [hasExited, setHasExited] = useState(false)
  
  // Use refs for cleanup to avoid stale closures
  const roomRef = useRef<Room | null>(null)
  const audioTrackRef = useRef<LocalAudioTrack | null>(null)
  const chatMessagesRef = useRef<HTMLDivElement | null>(null)
  const userNameRef = useRef<string>('')
  
  // Track transcription segments by ID to handle streaming updates
  const transcriptionSegmentsRef = useRef<Map<string, { text: string; speaker: string; timestamp: number; isAgent: boolean }>>(new Map())
  
  // Map participant identity to their display name (for agent person names)
  const participantNameMapRef = useRef<Map<string, string>>(new Map())
  
  // Map participant identity to their person details (backstory, topic, personality)
  const participantDetailsMapRef = useRef<Map<string, { backstory: string; topic: string; personality: string }>>(new Map())
  
  // Track handoff timestamps: participant identity -> timestamp when handoff occurred
  const handoffTimestampsRef = useRef<Map<string, number>>(new Map())
  
  // Keep refs in sync with state
  useEffect(() => {
    roomRef.current = room
  }, [room])
  
  useEffect(() => {
    audioTrackRef.current = localAudioTrack
  }, [localAudioTrack])
  
  useEffect(() => {
    userNameRef.current = userName
  }, [userName])

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (chatMessagesRef.current) {
      chatMessagesRef.current.scrollTop = chatMessagesRef.current.scrollHeight
    }
  }, [messages])

  const connectToRoom = useCallback(async () => {
    if (isConnecting || isConnected) {
      return
    }

    setIsConnecting(true)
    
    // Prevent multiple simultaneous connection attempts
    let currentRoom: Room | null = null
    
    try {
      const roomName = `voice-agent-${Date.now()}`
      const participantName = userName || 'user'  // Use user's name if provided

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
        participant?: Participant,
        publication?: TrackPublication
      ) => {
        
        segments.forEach((segment) => {
          if (!segment.text || !segment.text.trim()) {
            return
          }
          
          // Use participant.identity or participant.name - prefer name if available (more reliable for agent names)
          const isAgent = (participant as RemoteParticipant)?.isAgent ?? false
          
          // Use startTime (in milliseconds) for timestamp, fallback to firstReceivedTime
          const timestampMs = segment.startTime || segment.firstReceivedTime || Date.now()
          
          // Get speaker name - prioritize stored name (updated via data messages) over participant.name
          // IMPORTANT: If segment already exists, keep its original speaker name (don't change old messages)
          let speaker: string
          if (isAgent) {
            // Check if this segment already exists in our map
            const existingSegment = transcriptionSegmentsRef.current.get(segment.id)
            
            if (existingSegment) {
              // Segment already exists - keep its original speaker name (don't change old messages)
              speaker = existingSegment.speaker
            } else {
              // New segment - use the current stored name (which may have been updated by a handoff)
              const storedName = participant ? participantNameMapRef.current.get(participant.identity) : null
              
              // Also update the stored name if participant.name is available and we don't have a stored name yet
              if (!storedName && participant?.name) {
                participantNameMapRef.current.set(participant.identity, participant.name)
              }
              
              // Use stored name (from data messages) for new segments
              speaker = storedName || participant?.name || participant?.identity || 'Agent'
            }
          } else {
            // For user, use user's name if provided, otherwise fallback to local participant name or identity
            const userNameValue = userNameRef.current || newRoom.localParticipant?.name || newRoom.localParticipant?.identity || 'You'
            speaker = userNameValue
          }
          
          
          // Update or add segment to our tracking map
          // If segment ID already exists, it will be updated (for streaming)
          transcriptionSegmentsRef.current.set(segment.id, {
            text: segment.text,
            speaker: speaker,
            timestamp: timestampMs,
            isAgent: isAgent
          })
          
          // Rebuild transcription from all segments
          // Each unique segment ID represents one distinct message/utterance
          // Sort by timestamp to maintain chronological order
          const allSegments = Array.from(transcriptionSegmentsRef.current.entries())
            .map(([id, seg]) => ({ id, ...seg }))
            .sort((a, b) => a.timestamp - b.timestamp)
          
          // Group segments by their ID - each ID = one message
          // When same ID appears multiple times, it's a streaming update (we keep the latest)
          const messagesBySegmentId = new Map<string, { speaker: string; text: string; startTime: number; isAgent: boolean }>()
          
          allSegments.forEach((seg) => {
            // Each segment ID represents a distinct message
            // If we've seen this ID before, update it (streaming update)
            // Otherwise, create a new message
            if (!messagesBySegmentId.has(seg.id)) {
              messagesBySegmentId.set(seg.id, {
                speaker: seg.speaker,
                text: seg.text.trim(),
                startTime: seg.timestamp,
                isAgent: seg.isAgent
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
          
          // Convert to Message objects and sort by timestamp
          const messageObjects: Message[] = Array.from(messagesBySegmentId.entries())
            .map(([id, msg]) => ({
              id: id,
              speaker: msg.speaker,
              text: msg.text,
              timestamp: msg.startTime,
              isUser: !msg.isAgent
            }))
            .sort((a, b) => a.timestamp - b.timestamp)
          
          // Check if agent has spoken (first agent message)
          const agentHasSpoken = messageObjects.some(msg => !msg.isUser)
          if (agentHasSpoken && !hasAgentSpoken) {
            setHasAgentSpoken(true)
            // Unmute the audio track when agent first speaks
            if (audioTrackRef.current) {
              audioTrackRef.current.unmute()
            }
          }
          
          // Update messages state
          setMessages(messageObjects)
          
          // Track current speaker (most recent message)
          if (messageObjects.length > 0) {
            const mostRecentMessage = messageObjects[messageObjects.length - 1]
            setCurrentSpeaker(mostRecentMessage.speaker)
          }
          
          // Clean up finalized segments (optional - keep them for history)
          // if (segment.final) {
          //   // Segment is finalized, could clean up old segments if needed
          // }
        })
      })

      // Set up connection events
      newRoom.on(RoomEvent.Connected, async () => {
        
        // Request microphone access and publish audio track
        try {
          const audioTrack = await createLocalAudioTrack({
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          })
          // Mute the track initially - user should wait for agent to speak first
          await audioTrack.mute()
          await newRoom.localParticipant.publishTrack(audioTrack)
          setLocalAudioTrack(audioTrack)
        } catch (error) {
          alert('Failed to access microphone. Please check your permissions.')
          setIsConnecting(false)
          return
        }
        
        // Wait briefly for agent to connect (optional but improves UX)
        // The agent should connect within a few seconds, but we don't want to block too long
        const waitForAgent = new Promise<void>((resolve) => {
          const timeout = setTimeout(() => {
            resolve()
          }, 3000)  // Reduced to 3 seconds to avoid deadlock
          
          const checkForAgent = () => {
            // Check existing participants
            for (const participant of newRoom.remoteParticipants.values()) {
              if (participant.identity.includes('agent') || participant.isAgent) {
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
              clearTimeout(timeout)
              resolve()
            }
          })
        })
        
        await waitForAgent
        
        // Send user name to agent via data message
        if (userName) {
          const userNameMessage = JSON.stringify({
            type: 'user_name',
            name: userName
          })
          try {
            newRoom.localParticipant.publishData(
              new TextEncoder().encode(userNameMessage)
            )
          } catch (error) {
            // Failed to send user name
          }
        }
        
        // Don't disconnect if agent isn't found - just continue
        // The agent may connect later and the session will attach
        setIsConnected(true)
        setIsConnecting(false)
        setHasStarted(true)
        
        // Log current room state
        newRoom.remoteParticipants.forEach((participant) => {
          // Store names for existing participants (they may have connected before our listener was set up)
          if (participant.isAgent && participant.name) {
            participantNameMapRef.current.set(participant.identity, participant.name)
          }
        })
      })

      newRoom.on(RoomEvent.Disconnected, (reason) => {
        setIsConnected(false)
        setIsConnecting(false)
        setHasAgentSpoken(false)
        setCurrentSpeaker(null)
        // Clear transcription segments and name mappings on disconnect
        transcriptionSegmentsRef.current.clear()
        participantNameMapRef.current.clear()
        participantDetailsMapRef.current.clear()
        // Note: Messages are NOT cleared so they remain visible after stopping
      })



      newRoom.on(RoomEvent.MediaDevicesError, (error) => {
        // Media devices error
      })

      // Listen for data messages (e.g., person name from agent)
      newRoom.on(RoomEvent.DataReceived, (payload, participant, _kind, _topic) => {
        try {
          const data = new TextDecoder().decode(payload)
          const message = JSON.parse(data)
          
          if (message.type === 'person_name' && message.name && participant) {
            const oldName = participantNameMapRef.current.get(participant.identity)
            const handoffTimestamp = Date.now() // Record when handoff occurred
            
            // Store the handoff timestamp so we can identify which segments should use the new name
            if (oldName !== message.name) {
              handoffTimestampsRef.current.set(participant.identity, handoffTimestamp)
            }
            
            participantNameMapRef.current.set(participant.identity, message.name)
            
            // Store person details if provided
            if (message.backstory && message.topic && message.personality) {
              participantDetailsMapRef.current.set(participant.identity, {
                backstory: message.backstory,
                topic: message.topic,
                personality: message.personality
              })
            }
          }
        } catch (error) {
          // Not a JSON message or not the person_name type, ignore
        }
      })

      // Helper function to store participant name
      const storeParticipantName = (participant: RemoteParticipant) => {
        if (participant.name) {
          participantNameMapRef.current.set(participant.identity, participant.name)
        }
      }

      newRoom.on(RoomEvent.ParticipantConnected, (participant) => {
        // Store the participant's name for use in transcriptions
        // For agents, the name should be the person's name (e.g., "Avery Kim")
        storeParticipantName(participant)
      })

      newRoom.on(RoomEvent.TrackSubscribed, (
        track: Track,
        publication: RemoteTrackPublication,
        participant: RemoteParticipant
      ) => {
        if (track.kind === Track.Kind.Audio) {
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
          }
          
          track.attach(audioElement)
          
          audioElement.onerror = (error) => {
            // Audio playback error
          }
        }
      })

      // Connect to room
      try {
        await newRoom.connect(LIVEKIT_URL, token)
        
        // Room is already set above, but verify it's still the same instance
        if (currentRoom !== newRoom) {
          // Room instance changed during connection
        }
      } catch (connectError) {
        // Clean up on connection failure
        if (currentRoom) {
          currentRoom.disconnect().catch(() => {})
          setRoom(null)
        }
        alert('Failed to connect to room. Check your configuration.')
        setIsConnecting(false)
        return
      }
      
      // Monitor connection state
      newRoom.on(RoomEvent.ConnectionStateChanged, (state) => {
        if (state === 'disconnected') {
          // Room disconnected unexpectedly
        }
      })
    } catch (error) {
      // Clean up on error
      if (currentRoom) {
        currentRoom.disconnect().catch(() => {})
        setRoom(null)
      }
      alert('Failed to connect to room. Check your configuration.')
      setIsConnecting(false)
    }
    // Remove dependencies that cause re-renders - only depend on what's truly needed
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const disconnectFromRoom = useCallback(async () => {
    if (localAudioTrack) {
      localAudioTrack.stop()
      localAudioTrack.detach()
      setLocalAudioTrack(null)
    }
    if (room) {
      try {
        await room.disconnect()
      } catch (error) {
        // Error disconnecting room
      }
      setRoom(null)
      setIsConnected(false)
      setHasAgentSpoken(false)
      setCurrentSpeaker(null)
      setHasExited(true)
      // Clear transcription segments and name mappings on disconnect
      transcriptionSegmentsRef.current.clear()
      participantNameMapRef.current.clear()
      participantDetailsMapRef.current.clear()
    }
  }, [room, localAudioTrack])

  // Cleanup on unmount only - NOT on every room/track change
  // This prevents accidental disconnects during re-renders
  useEffect(() => {
    return () => {
      // Only cleanup on component unmount
      // Use refs to get current values, not stale closures
      const currentTrack = audioTrackRef.current
      const currentRoom = roomRef.current
      
      if (currentTrack) {
        currentTrack.stop()
        currentTrack.detach()
      }
      if (currentRoom) {
        currentRoom.disconnect().catch(() => {})
      }
    }
    // Empty dependency array = only run on unmount
  }, [])

  return (
    <div className="app">
      <div className="container">
        <h1>MockEvents</h1>
        <h2 className="subheading">Voice Agents Mixer: Streaming, Latency, Interrupt</h2>
        

        {/* Display agent details - show all agents that have connected */}
        {hasStarted && !hasExited && (() => {
          // Get all agent participants with their details
          const agentDetails: Array<{ identity: string; name: string; details: { backstory: string; topic: string; personality: string } }> = []
          
          for (const [identity, name] of participantNameMapRef.current.entries()) {
            const details = participantDetailsMapRef.current.get(identity)
            if (details && name && name !== userName && name !== 'You' && !name.toLowerCase().includes('user')) {
              agentDetails.push({ identity, name, details })
            }
          }
          
          // Show placeholder message if connected but no agent has spoken yet
          if (!hasAgentSpoken && agentDetails.length === 0) {
            return (
              <div className="speaker-details">
                <div className="speaker-details-placeholder">
                  <h3 className="speaker-details-title">Looking for people to connect with...</h3>
                  <p style={{ margin: 0, color: 'rgba(255, 255, 255, 0.9)', fontStyle: 'italic' }}>
                    Please wait while we match you with networking professionals. A person will approach you first - you will not approach them.
                  </p>
                </div>
              </div>
            )
          }
          
          // Show agent details when available
          if (agentDetails.length > 0) {
            return (
              <div className="speaker-details">
                <h3 className="speaker-details-title">Speaking with:</h3>
                {agentDetails.map(({ identity, name, details }) => (
                  <div key={identity} className="speaker-details-item">
                    <div className="speaker-details-header">
                      <h4 className="speaker-name">{name}</h4>
                      <div className="speaker-meta">
                        <span className="speaker-topic">{details.topic}</span>
                        <span className="speaker-personality">{details.personality}</span>
                      </div>
                    </div>
                    <div className="speaker-backstory">{details.backstory}</div>
                  </div>
                ))}
              </div>
            )
          }
          return null
        })()}

        {hasStarted && (
          <div className="chat-container">
            <h2>Transcription</h2>
            <div className="chat-messages" ref={chatMessagesRef}>
              {messages.length === 0 ? (
                <div className="chat-empty">Messages will appear here...</div>
              ) : (
                messages.map((message) => (
                  <MessageBubble key={message.id} message={message} />
                ))
              )}
            </div>
          </div>
        )}

        {hasExited ? (
          <div className="exit-message">
            <h2 style={{ color: '#4CAF50', marginBottom: '1rem' }}>Thanks for attending the event!</h2>
            <p style={{ color: 'rgba(255, 255, 255, 0.9)', fontStyle: 'italic' }}>
              We hope you enjoyed your networking experience.
            </p>
          </div>
        ) : (
          <>
            {!isConnected && (
              <>
                <div className="event-description">
                  <h3 className="event-description-title">About This Event</h3>
                  <p className="event-description-text">
                    Welcome to a networking mixer featuring three voice technology experts. 
                    Each specialist has deep knowledge in their field and a unique personality.
                  </p>
                  <div className="event-participants">
                    <div className="participant-card">
                      <div className="participant-name">Skye Morales</div>
                      <div className="participant-topic">Interruption Management</div>
                      <div className="participant-personality">Comedian</div>
                    </div>
                    <div className="participant-card">
                      <div className="participant-name">Noah Reed</div>
                      <div className="participant-topic">Latency Optimization</div>
                      <div className="participant-personality">Professional</div>
                    </div>
                    <div className="participant-card">
                      <div className="participant-name">Avery Kim</div>
                      <div className="participant-topic">Streaming Technology</div>
                      <div className="participant-personality">Aloof</div>
                    </div>
                  </div>
                  <p className="event-description-note">
                    <strong>How it works:</strong> Enter your name below and join the event. 
                    One of the experts will approach you first. You can ask questions about their 
                    expertise, and they'll naturally introduce you to others if needed. Conversations 
                    are transcribed in real-time.
                  </p>
                </div>
                <div className="user-name-input">
                  <label htmlFor="user-name">Your Name:</label>
                  <input
                    id="user-name"
                    type="text"
                    value={userName}
                    onChange={(e) => setUserName(e.target.value)}
                    placeholder="Enter your name"
                    disabled={isConnecting}
                  />
                </div>
              </>
            )}

            <div className="controls">
              {!isConnected ? (
                <button
                  onClick={connectToRoom}
                  disabled={isConnecting}
                  className="btn btn-start"
                >
                  {isConnecting ? 'Joining...' : 'Join Event'}
                </button>
              ) : (
                <button
                  onClick={disconnectFromRoom}
                  className="btn btn-stop"
                >
                  Exit Event
                </button>
              )}
            </div>
          </>
        )}

      </div>
    </div>
  )
}

export default App

