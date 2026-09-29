from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_milvus import Milvus
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model_config import embedding_model, model

vector_store = None

def _get_vector_store():
    global vector_store
    if vector_store is None:
        persist_dir = Path(__file__).parent / "milvus_db"
        persist_dir.mkdir(parents=True, exist_ok=True)
        vector_store = Milvus(
            embedding_function=embedding_model,
            connection_args={"uri": str(persist_dir / "milvus.db")},
            collection_name="aws-basics",
            index_params={"index_type": "FLAT", "metric_type": "COSINE"},
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


def answer_question(
    user_query: str,
    history: list | None = None,
    k: int = 5,
    expr: str | None = None,
) -> str:
    """Retrieve relevant chunks and answer only from those chunks."""
    if not user_query.strip():
        raise ValueError("user_query must not be empty")

    history = history or []
    query_embedding = embedding_model.embed_query(user_query)
    # print(f"\nQuery embedding generated for the question having length {len(query_embedding)}.")
    # This text-based call embeds the query inside the vector store.
    # retrieved_documents = similarity_search(user_query, k=k, expr=expr)
    retrieved_documents = _get_vector_store().similarity_search_by_vector(
        embedding=query_embedding,
        k=k,
        expr=expr,
    )
    context = "\n\n".join(
        f"Source {index}: {document.page_content}"
        for index, document in enumerate(retrieved_documents, start=1)
        if document.page_content.strip()
    )
    # print("\nContext retrieved for the question:\n", context, "\n\n")
    if not context:
        return "I don't have enough information in the provided content to answer that."

    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            *history[-6:],
            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION:\n{user_query}"),
        ]
    )
    return (
        response.content if isinstance(response.content, str) else str(response.content)
    )


def ingest_pdf(pdf_path: Path) -> None:
    """Load the PDF, chunk it and store embeddings. Skips work if already indexed."""
    store = _get_vector_store()

    if _get_vector_store().get_pks(expr="pk >= 0"):
        print("Existing index found - skipping ingestion.")
        return

    docs = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} pages and created {len(chunks)} chunks")

    ids = store.add_documents(chunks)
    print(f"Indexed {len(ids)} chunks")


if __name__ == "__main__":
    pdf_path = Path(__file__).resolve().parent.parent / "books" / "aws_basics.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    ingest_pdf(pdf_path)

    history: list = []
    while True:
        user_query = input("\nEnter your question (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        answer = answer_question(user_query, history=history)
        print("\nAnswer:", answer)
        history.extend([HumanMessage(content=user_query), AIMessage(content=answer)])
