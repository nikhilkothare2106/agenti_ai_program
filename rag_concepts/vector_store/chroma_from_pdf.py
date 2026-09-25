from dotenv import load_dotenv

from langchain_chroma import Chroma

from embedding_model_config import embeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

loader = PyPDFLoader(str(Path(__file__).parent.parent / "books" / "sample.pdf"))

docs = loader.load()
print(f"Loaded {len(docs)} pages")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(docs)
print(f"Loaded {len(docs)} pages and created {len(chunks)} chunks")

persist_dir = Path("rag_concepts/chroma_db")
collection_name = "sample-pdf"

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=str(persist_dir),
    collection_name=collection_name,
)

try:
    vector_store.reset_collection()
    print(f"Reset existing Chroma collection: {collection_name}")
except Exception as exc:
    print(f"Collection reset failed: {exc}")

# add documents
added_document_ids = vector_store.add_documents(chunks)
# print("Added document IDs:", added_document_ids)

# view documents
# documents_embeddings = vector_store.get(include=["embeddings"])
# print("Initial documents:", documents_embeddings)

# document_metadata = vector_store.get(include=["metadatas"])
# print("Document metadata:", document_metadata)

# document_content = vector_store.get(include=["documents"])
# print("Document content:", document_content)

stored_chunks = vector_store.get(include=["documents", "metadatas", "embeddings"])

for index, document_content in enumerate(stored_chunks["documents"]):
    metadata = stored_chunks["metadatas"][index]
    embedding = stored_chunks["embeddings"][index]
    print(f"Document chunk: {document_content}...")
    # print(f"Metadata: {metadata}")
    # print(f"Embedding length: {len(embedding)}")


# search documents
# similarity_search_results = vector_store.similarity_search(
#     query="What is machine learning?", k=2
# )
# print("Similarity search:", similarity_search_results)

# # search with similarity score
# similarity_search_with_score_results = vector_store.similarity_search_with_score(
#     query="How does artificial intelligence differ from machine learning?", k=2
# )
# print("Similarity search with score:", similarity_search_with_score_results)

# # meta-data filtering
# metadata_filtered_search_results = vector_store.similarity_search_with_score(
#     query="What are examples of artificial intelligence applications?",
#     filter={"page": 0},
# )
# print("Metadata-filtered search:", metadata_filtered_search_results)

# # update documents
# updated_doc1 = Document(
#     page_content="Artificial intelligence is the broader field of creating systems that perform tasks requiring human intelligence. Machine learning is a subset of AI that learns patterns from data to make predictions or decisions.",
#     metadata={"page": 0},
# )

# updated_document_result = vector_store.update_document(
#     document_id=added_document_ids[0], document=updated_doc1
# )
# print("Updated document:", updated_document_result)

# # view documents
# documents_after_update = vector_store.get(
#     include=["embeddings", "documents", "metadatas"]
# )
# print("After update:", documents_after_update)

# # delete document
# deleted_document_ids = vector_store.delete(ids=[added_document_ids[0]])
# print("Deleted document IDs:", deleted_document_ids)

# # view documents
# documents_after_delete = vector_store.get(
#     include=["embeddings", "documents", "metadatas"]
# )
# print("After delete:", documents_after_delete)

