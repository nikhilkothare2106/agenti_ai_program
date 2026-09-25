from pathlib import Path
from pypdf import PdfReader
import chromadb
from embedding_model_config import (
    client as embeddings,
)  # must now expose an .embed_documents(list[str]) -> list[list[float]] method, or adapt below

# --- 1. Load PDF pages (replaces PyPDFLoader) ---
pdf_path = Path(__file__).parent.parent / "books" / "sample.pdf"
reader = PdfReader(str(pdf_path))

docs = []  # list of dicts: {"content": str, "metadata": {...}}
for page_num, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    docs.append(
        {
            "content": text,
            "metadata": {"source": str(pdf_path), "page": page_num},
        }
    )

print(f"Loaded {len(docs)} pages")


# --- 2. Recursive character text splitter (replaces RecursiveCharacterTextSplitter) ---
def recursive_split(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200, separators=None
) -> list[str]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, seps: list[str]) -> list[str]:
        if len(text) <= chunk_size:
            return [text] if text else []

        if not seps:
            # hard split
            return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

        sep = seps[0]
        parts = text.split(sep) if sep else list(text)

        chunks = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size:
                    chunks.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(text, separators)

    # apply overlap
    if chunk_overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks

    overlapped = [raw_chunks[0]]
    for i in range(1, len(raw_chunks)):
        prev_tail = overlapped[-1][-chunk_overlap:]
        overlapped.append(prev_tail + raw_chunks[i])
    return overlapped


chunks = []  # list of dicts: {"content": str, "metadata": {...}}
for doc in docs:
    for piece in recursive_split(doc["content"], chunk_size=1000, chunk_overlap=200):
        chunks.append({"content": piece, "metadata": doc["metadata"]})

print(f"Loaded {len(docs)} pages and created {len(chunks)} chunks")


# --- 3. Chroma client (replaces langchain_chroma.Chroma) ---
persist_dir = Path("rag_concepts/vector_store/chroma_db")
collection_name = "sample-pdf"

client = chromadb.PersistentClient(path=str(persist_dir))

try:
    client.delete_collection(collection_name)
    print(f"Reset existing Chroma collection: {collection_name}")
except Exception as exc:
    print(f"Collection reset failed (may not have existed): {exc}")

collection = client.create_collection(name=collection_name)

# --- 4. Embed + add documents ---
texts = [c["content"] for c in chunks]
metadatas = [c["metadata"] for c in chunks]
ids = [f"chunk-{i}" for i in range(len(chunks))]

# Adjust this call to match your embeddings object's actual API.
# If `embeddings` is an OpenAIEmbeddings-style object it likely has .embed_documents()
vectors = embeddings.embed_documents(texts)

collection.add(
    ids=ids,
    documents=texts,
    metadatas=metadatas,
    embeddings=vectors,
)
added_document_ids = ids

# --- 5. View stored documents ---
stored_chunks = collection.get(include=["documents", "metadatas", "embeddings"])

for index, document_content in enumerate(stored_chunks["documents"]):
    metadata = stored_chunks["metadatas"][index]
    embedding = stored_chunks["embeddings"][index]
    print(f"Document chunk: {document_content}...")
    # print(f"Metadata: {metadata}")
    # print(f"Embedding length: {len(embedding)}")


# --- 6. Similarity search (replaces vector_store.similarity_search) ---
def similarity_search(query: str, k: int = 2, filter: dict | None = None):
    query_vector = embeddings.embed_documents([query])[0]
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        where=filter,
        include=["documents", "metadatas", "distances"],
    )
    return results


similarity_search_results = similarity_search("What is machine learning?", k=2)
print("Similarity search:", similarity_search_results)

# with distances (chroma returns distances, not similarity scores directly)
# similarity_search_with_score_results = similarity_search(
#     "How does artificial intelligence differ from machine learning?", k=2
# )
# print("Similarity search with score:", similarity_search_with_score_results)

# metadata filtering
# metadata_filtered_search_results = similarity_search(
#     "What are examples of artificial intelligence applications?",
#     filter={"page": 0},
# )
# print("Metadata-filtered search:", metadata_filtered_search_results)


# --- 7. Update a document (replaces vector_store.update_document) ---
def update_document(doc_id: str, content: str, metadata: dict):
    vector = embeddings.embed_documents([content])[0]
    collection.update(
        ids=[doc_id],
        documents=[content],
        metadatas=[metadata],
        embeddings=[vector],
    )


updated_document_result = update_document(
    added_document_ids[0],
    "Artificial intelligence is the broader field of creating systems that perform "
    "tasks requiring human intelligence. Machine learning is a subset of AI that "
    "learns patterns from data to make predictions or decisions.",
    {"page": 0},
)

documents_after_update = collection.get(
    include=["embeddings", "documents", "metadatas"]
)
print("After update:", documents_after_update)


# --- 8. Delete a document (replaces vector_store.delete) ---
# collection.delete(ids=[added_document_ids[0]])
# print("Deleted document IDs:", [added_document_ids[0]])

# documents_after_delete = collection.get(include=["embeddings", "documents", "metadatas"])
# print("After delete:", documents_after_delete)
