const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

/**
 * Streams a chat reply from POST /api/chat/stream.
 *
 * The backend sends SSE frames: "event: <name>\ndata: <line>\n...\n\n".
 * event types used: "session" (the resolved session id, sent once),
 * "message" (default, a chunk of reply text), "error", "done".
 *
 * @param {string} sessionId
 * @param {string} message
 * @param {{ onChunk?: (text: string) => void, onSession?: (id: string) => void, signal?: AbortSignal }} handlers
 * @returns {Promise<void>} resolves once the stream ends
 */
export async function streamChat(sessionId, message, { onChunk, onSession, signal } = {}) {
  const res = await fetch(`${API_BASE_URL}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message }),
    signal,
  })

  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `Request failed with status ${res.status}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const handleFrame = (frame) => {
    let event = 'message'
    const dataLines = []
    for (const rawLine of frame.split('\n')) {
      if (rawLine.startsWith('event: ')) {
        event = rawLine.slice('event: '.length)
      } else if (rawLine.startsWith('data: ')) {
        dataLines.push(rawLine.slice('data: '.length))
      }
    }
    const data = dataLines.join('\n')

    if (event === 'session') {
      onSession?.(data)
    } else if (event === 'error') {
      throw new Error(data)
    } else if (event === 'done') {
      // no-op, stream is about to close
    } else {
      onChunk?.(data)
    }
  }

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const frame = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      if (frame.trim()) handleFrame(frame)
    }
  }
}
