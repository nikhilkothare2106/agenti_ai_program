export default function Sidebar({ threads, activeThreadId, onSelectThread, onNewChat }) {
  const ordered = [...threads].reverse()

  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <p className="sidebar__brand-title">AgentCore Chat</p>
        <p className="sidebar__brand-sub">bedrock-agentcore runtime</p>
      </div>

      <button className="sidebar__new" onClick={onNewChat}>
        + New chat
      </button>

      <p className="sidebar__section-label">Sessions</p>

      {ordered.length === 0 && (
        <p className="sidebar__empty">No sessions yet. Send a message to start one.</p>
      )}

      <div className="sidebar__threads">
        {ordered.map((thread) => (
          <button
            key={thread.id}
            className={
              'thread-btn' + (thread.id === activeThreadId ? ' thread-btn--active' : '')
            }
            onClick={() => onSelectThread(thread.id)}
          >
            <span className="thread-btn__id">{thread.id}</span>
            <span className="thread-btn__preview">
              {thread.preview || 'Empty session'}
            </span>
          </button>
        ))}
      </div>
    </aside>
  )
}
