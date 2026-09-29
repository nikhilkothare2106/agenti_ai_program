import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

DB_PATH = Path(__file__).parent / "memory.db"


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_memory() -> None:
    with _db() as db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL DEFAULT (strftime('%s','now'))
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id, id);

            CREATE TABLE IF NOT EXISTS summaries (
                session_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                upto_id INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS user_memory (
                user_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at REAL DEFAULT (strftime('%s','now')),
                PRIMARY KEY (user_id, key)
            );
            """
        )


# ---------- short-term: history + rolling summary ----------

def save_message(session_id: str, role: str, content: str) -> None:
    with _db() as db:
        db.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )


def load_context(session_id: str) -> tuple[str, list]:
    """Return (summary of older turns, unsummarized recent messages)."""
    with _db() as db:
        row = db.execute(
            "SELECT summary, upto_id FROM summaries WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        summary, upto = (row["summary"], row["upto_id"]) if row else ("", 0)
        rows = db.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id = ? AND id > ? ORDER BY id",
            (session_id, upto),
        ).fetchall()
    messages = [
        HumanMessage(content=r["content"]) if r["role"] == "user"
        else AIMessage(content=r["content"])
        for r in rows
    ]
    return summary, messages


def maybe_summarize(session_id: str, llm, keep: int = 6, trigger: int = 12) -> None:
    """Once >= trigger unsummarized messages exist, fold all but the last
    `keep` into the rolling summary."""
    with _db() as db:
        row = db.execute(
            "SELECT summary, upto_id FROM summaries WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        summary, upto = (row["summary"], row["upto_id"]) if row else ("", 0)
        rows = db.execute(
            "SELECT id, role, content FROM messages "
            "WHERE session_id = ? AND id > ? ORDER BY id",
            (session_id, upto),
        ).fetchall()

    if len(rows) < trigger:
        return

    old = rows[:-keep]
    transcript = "\n".join(f"{r['role']}: {r['content']}" for r in old)
    resp = llm.invoke(
        [
            SystemMessage(
                content="Update the running summary of this conversation. Keep "
                "topics discussed, what the user was trying to find out, and "
                "any answers already given. Max 150 words."
            ),
            HumanMessage(
                content=f"CURRENT SUMMARY:\n{summary or '(none)'}\n\n"
                f"NEW MESSAGES:\n{transcript}"
            ),
        ]
    )
    new_summary = resp.content if isinstance(resp.content, str) else str(resp.content)
    with _db() as db:
        db.execute(
            "INSERT INTO summaries (session_id, summary, upto_id) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "summary = excluded.summary, upto_id = excluded.upto_id",
            (session_id, new_summary, old[-1]["id"]),
        )


def condense_question(llm, query: str, summary: str, history: list) -> str:
    """Rewrite a follow-up into a standalone question for vector search."""
    if not summary and not history:
        return query
    convo = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in history[-4:]
    )
    resp = llm.invoke(
        [
            SystemMessage(
                content="Rewrite the user's latest question as a standalone "
                "search query, resolving pronouns and references using the "
                "conversation. If it is already standalone, return it "
                "unchanged. Output only the query."
            ),
            HumanMessage(
                content=f"SUMMARY:\n{summary or '(none)'}\n\n"
                f"RECENT:\n{convo}\n\nLATEST QUESTION:\n{query}"
            ),
        ]
    )
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return text.strip() or query


# ---------- long-term: user preferences ----------

def get_user_memory(user_id: str) -> dict[str, str]:
    with _db() as db:
        rows = db.execute(
            "SELECT key, value FROM user_memory WHERE user_id = ?", (user_id,)
        ).fetchall()
    return {r["key"]: r["value"] for r in rows}


def upsert_user_memory(user_id: str, facts: dict[str, str]) -> None:
    with _db() as db:
        for key, value in facts.items():
            db.execute(
                "INSERT INTO user_memory (user_id, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id, key) DO UPDATE SET "
                "value = excluded.value, updated_at = strftime('%s','now')",
                (user_id, key, value),
            )


def forget(user_id: str, key: str | None = None) -> None:
    with _db() as db:
        if key:
            db.execute(
                "DELETE FROM user_memory WHERE user_id = ? AND key = ?",
                (user_id, key),
            )
        else:
            db.execute("DELETE FROM user_memory WHERE user_id = ?", (user_id,))


EXTRACT_PROMPT = """
From the user's message, extract durable facts about the USER or how they want
answers delivered (e.g. answer_style, role, tools_used, language).

Rules:
- Only facts the user stated. Never guess.
- Never extract facts about the document's content.
- Skip one-off requests, health, financial, or other sensitive personal data.
- Return a flat JSON object of short snake_case keys to short string values.
- If nothing qualifies, return {}. Output JSON only.
""".strip()


def extract_user_memory(llm, user_query: str) -> dict[str, str]:
    resp = llm.invoke(
        [SystemMessage(content=EXTRACT_PROMPT), HumanMessage(content=user_query)]
    )
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    text = text.replace("```json", "").replace("```", "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(k)[:40]: str(v)[:200]
        for k, v in list(data.items())[:5]
        if isinstance(v, (str, int, float))
    }