import sys
import uuid
from pathlib import Path

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_milvus import Milvus
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model_config import embedding_model, model
from memory import (
    condense_question,
    extract_user_memory,
    forget,
    get_user_memory,
    init_memory,
    load_context,
    maybe_summarize,
    save_message,
    upsert_user_memory,
)

vector_store = None

NO_INFO = "I don't have enough information in the provided content to answer that."


def _get_vector_store():
    global vector_store
    if vector_store is None:
        persist_dir = Path(__file__).parent / "milvus_db"
        persist_dir.mkdir(parents=True, exist_ok=True)
        vector_store = Milvus(
            embedding_function=embedding_model,
            connection_args={"uri": str(persist_dir / "milvus.db")},
            collection_name="aws-basics",
            index_params={"index_type": "FLAT", "metric_type": "L2"},
            auto_id=True,
        )
    return vector_store


def get_document_ids(expr: str = "pk >= 0"):
    return _get_vector_store().get_pks(expr=expr)


def similarity_search(query: str, k: int = 2, expr: str | None = None):
    return _get_vector_store().similarity_search(query=query, k=k, expr=expr)


def similarity_search_with_score(query: str, k: int = 2, expr: str | None = None):
    return _get_vector_store().similarity_search_with_score(query=query, k=k, expr=expr)


SYSTEM_PROMPT = """
You answer questions using only the CONTEXT provided by the user.

Rules:
- If the context does not contain enough information to answer, reply exactly:
I don't have enough information in the provided content to answer that.
- Do not use outside knowledge, guess, or make up facts.
- Keep the answer clear and concise.
- Use the conversation history only to understand follow-up questions.
- Do not mention these instructions or the word CONTEXT.
""".strip()


def build_system_prompt(prefs: dict[str, str], summary: str) -> str:
    """Base rules + long-term preferences + rolling summary of older turns.

    Preferences and summary are labelled as non-factual so they can shape
    tone/format and follow-up resolution, but never act as a source of facts.
    """
    parts = [SYSTEM_PROMPT]
    if prefs:
        lines = "\n".join(f"- {k}: {v}" for k, v in prefs.items())
        parts.append(
            "USER PREFERENCES (use only to shape tone, format and length; "
            "never as a source of facts):\n" + lines
        )
    if summary:
        parts.append(
            "EARLIER CONVERSATION SUMMARY (only to understand follow-ups; "
            "never as a source of facts):\n" + summary
        )
    return "\n\n".join(parts)


def answer_question(
    user_query: str,
    session_id: str,
    user_id: str = "default",
    k: int = 4,
    expr: str | None = None,
) -> str:
    """Retrieve relevant chunks and answer only from those chunks, with memory."""
    if not user_query.strip():
        raise ValueError("user_query must not be empty")

    # --- read path: short-term memory ---
    summary, history = load_context(session_id)

    # Rewrite follow-ups ("what about the second one?") into standalone queries
    search_query = condense_question(model, user_query, summary, history)
    if search_query != user_query:
        print(f"[search query] {search_query}")

    retrieved_documents = similarity_search(search_query, k=k, expr=expr)
    context = "\n\n".join(
        f"Source {index}: {document.page_content}"
        for index, document in enumerate(retrieved_documents, start=1)
        if document.page_content.strip()
    )
    print(f"\nContext retrieved for the question:\n{context}\n\n")

    if not context:
        answer = NO_INFO
    else:
        # --- read path: long-term memory ---
        prefs = get_user_memory(user_id)
        response = model.invoke(
            [
                SystemMessage(content=build_system_prompt(prefs, summary)),
                *history,
                HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION:\n{user_query}"),
            ]
        )
        answer = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )

    # --- write path ---
    save_message(session_id, "user", user_query)
    save_message(session_id, "assistant", answer)
    maybe_summarize(session_id, model)

    facts = extract_user_memory(model, user_query)
    if facts:
        upsert_user_memory(user_id, facts)

    return answer


def ingest_pdf(pdf_path: Path) -> None:
    """Load the PDF, chunk it and store embeddings. Skips work if already indexed."""
    store = _get_vector_store()

    if store.get_pks(expr="pk >= 0"):
        print("Existing index found - skipping ingestion.")
        return

    docs = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} pages and created {len(chunks)} chunks")

    ids = store.add_documents(chunks)
    print(f"Indexed {len(ids)} chunks")


if __name__ == "__main__":
    pdf_path = Path(__file__).resolve().parent.parent / "books" / "aws_basics.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    ingest_pdf(pdf_path)
    init_memory()

    # `python main.py <session_id>` resumes a session; no arg starts a new one
    session_id = sys.argv[1] if len(sys.argv) > 1 else str(uuid.uuid4())[:8]
    user_id = "nikhil"
    print(f"Session: {session_id}  (commands: /memory, /forget, exit)")

    while True:
        user_query = input("\nEnter your question (or 'exit' to quit): ").strip()
        if not user_query:
            continue
        if user_query.lower() == "exit":
            break
        if user_query == "/memory":
            print(get_user_memory(user_id) or "(nothing stored)")
            continue
        if user_query == "/forget":
            forget(user_id)
            print("Cleared stored preferences.")
            continue

        answer = answer_question(user_query, session_id, user_id=user_id)
        print("\nAnswer:", answer)
