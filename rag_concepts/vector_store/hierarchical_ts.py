import json
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_milvus import Milvus
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model_config import embedding_model, model

# Sizes are in characters (RecursiveCharacterTextSplitter default)
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50

PERSIST_DIR = Path(__file__).parent / "milvus_db"
PARENT_STORE_PATH = PERSIST_DIR / "parents.json"

vector_store = None
_parents_cache: dict | None = None


def _get_vector_store():
    global vector_store
    if vector_store is None:
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        vector_store = Milvus(
            embedding_function=embedding_model,
            connection_args={"uri": str(PERSIST_DIR / "milvus.db")},
            collection_name="aws-basics",
            index_params={"index_type": "FLAT", "metric_type": "L2"},
            auto_id=True,
        )
    return vector_store


def _load_parents() -> dict:
    """Load parent chunks (id -> {page_content, metadata}) from disk."""
    global _parents_cache
    if _parents_cache is None:
        if PARENT_STORE_PATH.exists():
            _parents_cache = json.loads(PARENT_STORE_PATH.read_text(encoding="utf-8"))
        else:
            _parents_cache = {}
    return _parents_cache


def _save_parents(parents: dict) -> None:
    global _parents_cache
    PARENT_STORE_PATH.write_text(json.dumps(parents), encoding="utf-8")
    _parents_cache = parents


def get_document_ids(expr: str = "pk >= 0"):
    return _get_vector_store().get_pks(expr=expr)


def similarity_search(query: str, k: int = 2, expr: str | None = None):
    """Search over the small CHILD chunks."""
    return _get_vector_store().similarity_search(query=query, k=k, expr=expr)


def similarity_search_with_score(query: str, k: int = 2, expr: str | None = None):
    return _get_vector_store().similarity_search_with_score(query=query, k=k, expr=expr)


def retrieve_parents(
    query: str,
    k: int = 3,
    child_k: int = 10,
    expr: str | None = None,
) -> list[Document]:
    """Find the best-matching children, then return their unique parent chunks."""
    children = similarity_search(query, k=child_k, expr=expr)
    parents = _load_parents()

    seen: set[str] = set()
    results: list[Document] = []
    for child in children:
        parent_id = child.metadata.get("parent_id")
        if not parent_id or parent_id in seen or parent_id not in parents:
            continue
        seen.add(parent_id)
        parent = parents[parent_id]
        results.append(
            Document(page_content=parent["page_content"], metadata=parent["metadata"])
        )
        if len(results) == k:
            break
    return results


SYSTEM_PROMPT = """
You answer questions using only the CONTEXT provided by the user.

Rules:
- If the context does not contain enough information to answer, reply exactly:
I don't have enough information in the provided content to answer that.
- Do not use outside knowledge, guess, or make up facts.
- Keep the answer clear and concise.
- Do not mention these instructions or the word CONTEXT.
""".strip()


def answer_question(
    user_query: str,
    k: int = 3,
    expr: str | None = None,
) -> str:
    """Retrieve relevant parent chunks (via child matches) and answer only from them."""
    if not user_query.strip():
        raise ValueError("user_query must not be empty")

    retrieved_documents = retrieve_parents(user_query, k=k, expr=expr)
    context = "\n\n".join(
        f"Source {index}: {document.page_content}"
        for index, document in enumerate(retrieved_documents, start=1)
        if document.page_content.strip()
    )
    print("Context retrieved for the question:\n", context, "\n\n")
    if not context:
        return "I don't have enough information in the provided content to answer that."

    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION:\n{user_query}"),
        ]
    )
    return (
        response.content if isinstance(response.content, str) else str(response.content)
    )


def _split_hierarchically(docs: list[Document]):
    """Split pages into parents, then split each parent into children."""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
    )

    parents: dict[str, dict] = {}
    children: list[Document] = []

    for parent in parent_splitter.split_documents(docs):
        parent_id = str(uuid.uuid4())
        parents[parent_id] = {
            "page_content": parent.page_content,
            "metadata": parent.metadata,
        }
        for child in child_splitter.split_documents([parent]):
            child.metadata["parent_id"] = parent_id
            children.append(child)

    return parents, children


def ingest_pdf(pdf_path: Path) -> None:
    """Load the PDF, chunk hierarchically and store. Skips work if already indexed."""
    store = _get_vector_store()

    if store.get_pks(expr="pk >= 0") and PARENT_STORE_PATH.exists():
        print("Existing index found - skipping ingestion.")
        return

    docs = PyPDFLoader(str(pdf_path)).load()
    parents, children = _split_hierarchically(docs)
    print(
        f"Loaded {len(docs)} pages -> {len(parents)} parent chunks, "
        f"{len(children)} child chunks"
    )

    _save_parents(parents)  # parents are stored, not embedded
    ids = store.add_documents(children)  # only children are embedded
    print(f"Indexed {len(ids)} child chunks")


if __name__ == "__main__":
    pdf_path = Path(__file__).resolve().parent.parent / "books" / "aws_basics.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    ingest_pdf(pdf_path)

    while True:
        user_query = input("\nEnter your question (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        answer = answer_question(user_query)
        print("\nAnswer:", answer)