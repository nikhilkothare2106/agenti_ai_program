from pathlib import Path
from functools import lru_cache

from langchain_community.document_loaders import PyPDFLoader
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model_config import embedding_model

BASE_DIR = Path(__file__).resolve().parent
PERSIST_DIR = BASE_DIR / "milvus_db"
COLLECTION_NAME = "employee-handbook"
PDF_PATH = BASE_DIR.parent / "books" / "employee_handbook.pdf"


def split_documents(documents, chunk_size: int = 600, chunk_overlap: int = 100):
    """Split loaded documents into overlapping chunks, preserving metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


@lru_cache(maxsize=1)
def get_vector_store():
    """Open the persistent local Milvus Lite collection."""
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return Milvus(
        embedding_function=embedding_model,
        connection_args={"uri": str(PERSIST_DIR / "milvus.db")},
        collection_name=COLLECTION_NAME,
        index_params={"index_type": "FLAT", "metric_type": "L2"},
        auto_id=True,
    )


def embed(
    pdf_path: str | Path,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
):
    """Load, chunk, and index a PDF"""
    if not pdf_path.exists():
        raise FileNotFoundError(f"Handbook PDF not found: {pdf_path}")

    vector_store = get_vector_store()
    col = getattr(vector_store, "col", None)
    try:
        if col is not None and getattr(col, "num_entities", 0) > 0:
            print("Vector Embedding done already!")
            return
    except Exception:
        pass

    documents = PyPDFLoader(str(pdf_path)).load()
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError(f"No text was extracted from PDF: {pdf_path}")

    ids = vector_store.add_documents(chunks)
    print(f"Loaded {len(documents)} pages and created {len(chunks)} chunks")
    print(f"Indexed {len(ids)} chunks in '{COLLECTION_NAME}'")


if __name__ == "__main__":
    embed(PDF_PATH)
