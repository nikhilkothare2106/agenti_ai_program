import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

function MessageBubble({ msg }) {
  if (msg.role === 'user') {
    return <div className="msg msg--user">{msg.content}</div>
  }

  const variant = msg.error ? 'error' : 'assistant'
  return (
    <div className={`msg msg--${variant}`}>
      <div className="markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {msg.content || (msg.streaming ? '' : '')}
        </ReactMarkdown>
      </div>
      {msg.streaming && <span className="msg__cursor" aria-hidden="true" />}
    </div>
  )
}

export default function ChatWindow({ sessionId, messages, onSend, isSending }) {
  const [input, setInput] = useState('')
  const scrollRef = useRef(null)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  const handleSubmit = (e) => {
    e.preventDefault()
    const trimmed = input.trim()
    if (!trimmed || isSending) return
    onSend(trimmed)
    setInput('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSubmit(e)
    }
  }

  return (
    <main className="main">
      <header className="main__header">
        <span className="main__header-title">Conversation</span>
        <span className="session-tag">{sessionId}</span>
      </header>

      <div className="messages" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="messages__empty">
            <p className="messages__empty-title">Start the runtime</p>
            <p>Send a message below to invoke the agent for this session.</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <MessageBubble key={i} msg={msg} />
        ))}
      </div>

      <form className="composer" onSubmit={handleSubmit}>
        <textarea
          className="composer__input"
          rows={1}
          placeholder="Type here"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button className="composer__send" type="submit" disabled={isSending || !input.trim()}>
          Send
        </button>
      </form>
    </main>
  )
}
