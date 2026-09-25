from dotenv import load_dotenv

from langchain_milvus import Milvus
from langchain_core.documents import Document

from embedding_model_config import client
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

# loader = PyPDFLoader(str(Path(__file__).parent.parent / "books" / "sample.pdf"))
loader = PyPDFLoader(str(Path(__file__).parent.parent / "books" / "aws_basics.pdf"))

docs = loader.load()
print(f"Loaded {len(docs)} pages")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)
print(f"Loaded {len(docs)} pages and created {len(chunks)} chunks")


persist_dir = Path(__file__).parent / "milvus_db"
persist_dir.mkdir(parents=True, exist_ok=True)
collection_name = "aws-basics"

vector_store = Milvus(
    embedding_function=client,
    connection_args={"uri": str(persist_dir / "milvus.db")},
    collection_name=collection_name,
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    auto_id=True,
)


def get_document_ids(expr: str = "pk >= 0"):
    return vector_store.get_pks(expr=expr)


def similarity_search(query: str, k: int = 2, expr: str | None = None):
    return vector_store.similarity_search(query=query, k=k, expr=expr)


def similarity_search_with_score(query: str, k: int = 2, expr: str | None = None):
    return vector_store.similarity_search_with_score(query=query, k=k, expr=expr)


# add documents
added_document_ids = vector_store.add_documents(chunks)
print("Added document IDs:", added_document_ids)


# # view stored document IDs
# stored_document_ids = get_document_ids()
# print("Initial document IDs:", stored_document_ids)


# search documents
similarity_search_results = similarity_search("What is machine learning?", k=2)
print("Similarity search:", similarity_search_results)

# search with similarity score
similarity_search_with_score_results = similarity_search_with_score(
    "How does artificial intelligence differ from machine learning?", k=2
)
print("Similarity search with score:", similarity_search_with_score_results, "\n\n")

# meta-data filtering
metadata_filtered_search_results = similarity_search_with_score(
    "What are examples of artificial intelligence applications?",
    expr="page == 0",
)
print("Metadata-filtered search:", metadata_filtered_search_results, "\n\n")

# update documents
updated_doc1 = Document(
    page_content="Artificial intelligence is the broader field of creating systems that perform tasks requiring human intelligence. Machine learning is a subset of AI that learns patterns from data to make predictions or decisions.",
    metadata=chunks[0].metadata,
)

vector_store.upsert(ids=[added_document_ids[0]], documents=[updated_doc1])
print("Updated document ID:", added_document_ids[0])

# view document IDs after update
document_ids_after_update = get_document_ids()
print("After update:", document_ids_after_update)

# delete document
deleted_document_ids = vector_store.delete(ids=[added_document_ids[0]])
print("Deleted document IDs:", deleted_document_ids)

# view document IDs after delete
document_ids_after_delete = get_document_ids()
print("After delete:", document_ids_after_delete)
