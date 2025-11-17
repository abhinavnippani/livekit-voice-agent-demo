/**
 * Token Server Helper
 * 
 * In production, you should generate tokens on your backend server.
 * This is a helper function that you can use to call your backend.
 */

export async function generateToken(
  roomName: string,
  participantName: string,
  livekitUrl: string,
  apiKey: string,
  apiSecret: string
): Promise<string> {
  // Option 1: Call your backend API
  try {
    const response = await fetch('/api/token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        roomName,
        participantName,
      }),
    })

    if (response.ok) {
      const data = await response.json()
      return data.token
    }
  } catch (error) {
    console.warn('Backend token endpoint not available, using client-side generation')
  }

  // Option 2: Generate token client-side (for development only)
  // Note: This requires exposing your API secret, which is NOT secure for production
  // Only use this for development/testing
  if (import.meta.env.DEV && apiSecret) {
    const { AccessToken } = await import('livekit-client')
    const token = new AccessToken(apiKey, apiSecret, {
      identity: participantName,
    })
    token.addGrant({
      room: roomName,
      roomJoin: true,
      canPublish: true,
      canSubscribe: true,
      canPublishData: true,
    })
    return await token.toJwt()
  }

  throw new Error('Token generation failed. Please set up a backend token server.')
}

