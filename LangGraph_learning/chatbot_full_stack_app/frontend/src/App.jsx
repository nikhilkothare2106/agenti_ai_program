import { useEffect, useState } from 'react'
import Sidebar from './components/Sidebar.jsx'
import ChatWindow from './components/ChatWindow.jsx'
import { streamChat } from './api.js'

const STORAGE_KEY = 'agentcore-chat-threads'

// runtimeSessionId must be 33+ characters
function generateSessionId() {
  return crypto.randomUUID() + crypto.randomUUID().slice(0, 4)
}

function loadThreads() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

export default function App() {
  const [threads, setThreads] = useState(loadThreads)
  const [activeThreadId, setActiveThreadId] = useState(() => threads[0]?.id || generateSessionId())
  const [isSending, setIsSending] = useState(false)

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(threads))
  }, [threads])

  const activeThread = threads.find((t) => t.id === activeThreadId)
  const messages = activeThread?.messages || []

  const ensureThreadExists = (id) => {
    setThreads((prev) => (prev.some((t) => t.id === id) ? prev : [...prev, { id, messages: [] }]))
  }

  const handleNewChat = () => {
    const id = generateSessionId()
    setThreads((prev) => [...prev, { id, messages: [] }])
    setActiveThreadId(id)
  }

  const handleSelectThread = (id) => {
    setActiveThreadId(id)
  }

  const appendMessage = (threadId, message) => {
    setThreads((prev) =>
      prev.map((t) => (t.id === threadId ? { ...t, messages: [...t.messages, message] } : t))
    )
  }

  // Mutates the last message of a thread in place - used while a reply streams in.
  const updateLastMessage = (threadId, updater) => {
    setThreads((prev) =>
      prev.map((t) => {
        if (t.id !== threadId) return t
        const msgs = t.messages
        if (msgs.length === 0) return t
        const last = msgs[msgs.length - 1]
        return { ...t, messages: [...msgs.slice(0, -1), updater(last)] }
      })
    )
  }

  const handleSend = async (text) => {
    const threadId = activeThreadId
    ensureThreadExists(threadId)
    appendMessage(threadId, { role: 'user', content: text })
    appendMessage(threadId, { role: 'assistant', content: '', streaming: true })
    setIsSending(true)

    try {
      await streamChat(threadId, text, {
        onChunk: (chunk) => {
          updateLastMessage(threadId, (last) => ({ ...last, content: last.content + chunk }))
        },
      })
      updateLastMessage(threadId, (last) => ({ ...last, streaming: false }))
    } catch (err) {
      updateLastMessage(threadId, (last) => ({
        ...last,
        content: last.content || `Request failed: ${err.message}`,
        streaming: false,
        error: true,
      }))
    } finally {
      setIsSending(false)
    }
  }

  return (
    <div className="app">
      <Sidebar
        threads={threads.map((t) => ({
          id: t.id,
          preview: t.messages.find((m) => m.role === 'user')?.content,
        }))}
        activeThreadId={activeThreadId}
        onSelectThread={handleSelectThread}
        onNewChat={handleNewChat}
      />
      <ChatWindow
        sessionId={activeThreadId}
        messages={messages}
        onSend={handleSend}
        isSending={isSending}
      />
    </div>
  )
}
