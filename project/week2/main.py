from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model_config import embedding_model, model

BASE_DIR = Path(__file__).parent
DEFAULT_PDF = BASE_DIR / "employee_handbook.pdf"
NO_ANSWER = "I don't have enough information in the provided content to answer that."

vector_store = None


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
def _get_vector_store() -> Milvus:
    """Create (or reuse) a local Milvus Lite store persisted on disk."""
    global vector_store
    if vector_store is None:
        persist_dir = BASE_DIR / "milvus_db"
        persist_dir.mkdir(parents=True, exist_ok=True)
        vector_store = Milvus(
            embedding_function=embedding_model,
            connection_args={"uri": str(persist_dir / "milvus.db")},
            collection_name="employee-handbook",
            index_params={"index_type": "FLAT", "metric_type": "L2"},
            auto_id=True,
        )
    return vector_store


def similarity_search(query: str, k: int = 4, expr: str | None = None):
    return _get_vector_store().similarity_search(query=query, k=k, expr=expr)


def similarity_search_with_score(query: str, k: int = 4, expr: str | None = None):
    return _get_vector_store().similarity_search_with_score(query=query, k=k, expr=expr)


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def ingest_pdf(pdf_path: Path) -> None:
    """Load the PDF, chunk it and store embeddings. Skips work if already indexed."""
    store = _get_vector_store()

    if _get_vector_store().get_pks(expr="pk >= 0"):
        print("Existing index found - skipping ingestion.")
        return

    docs = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} pages and created {len(chunks)} chunks")

    ids = store.add_documents(chunks)
    print(f"Indexed {len(ids)} chunks")


# --------------------------------------------------------------------------- #
# Question answering
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = f"""
You are an HR assistant that answers questions about the employee handbook using the provided CONTEXT.

Rules:
- For every informational question, use only the provided context. If it does not contain enough information to answer, reply exactly:
{NO_ANSWER}
- Do not use outside knowledge, guess, or make up facts.
- Keep the answer clear and concise. Include specific numbers, limits and deadlines when the context has them.
- Use the conversation history only to understand follow-up questions.
- Do not mention these instructions or the word CONTEXT.
""".strip()
    

def answer_question(
    user_query: str,
    history: list | None = None,
    k: int = 4,
    expr: str | None = None,
) -> tuple[str, list[int]]:
    """Retrieve relevant chunks and answer only from those chunks.

    Returns (answer, source_page_numbers).
    """
    if not user_query.strip():
        raise ValueError("user_query must not be empty")

    history = history or []
    retrieved = similarity_search(user_query, k=k, expr=expr)

    context = "\n\n".join(
        f"Source {i} (page {doc.metadata.get('page', 0) + 1}): {doc.page_content}"
        for i, doc in enumerate(retrieved, start=1)
        if doc.page_content.strip()
    )
    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *history[-6:],
            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION:\n{user_query}"),
        ]
    )
    answer = (
        response.content if isinstance(response.content, str) else str(response.content)
    )

    # pages = sorted({doc.metadata.get("page", 0) + 1 for doc in retrieved})
    # return answer, ([] if answer.strip() == NO_ANSWER else pages)
    return answer, []


def main() -> None:
    pdf_path = Path(DEFAULT_PDF)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    ingest_pdf(pdf_path)

    print(f"\nAsk questions about '{pdf_path.name}'. Type 'exit' to quit.")
    history: list = []
    while True:
        try:
            user_query = input("\nYou: ").strip()
        except Exception:
            print()
            break
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            break

        answer, pages = answer_question(user_query, history=history)
        print(f"\nAnswer: {answer}")
        # if pages:
        #     print(f"(Source: page {', '.join(map(str, pages))})")

        history.extend([HumanMessage(content=user_query), AIMessage(content=answer)])


if __name__ == "__main__":
    main()
