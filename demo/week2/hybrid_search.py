import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from model_config import embedding_model, model

load_dotenv()

PDF_PATH = Path(__file__).parent.parent / "book" / "employee_handbook.pdf"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 5
NOT_FOUND = "I could not find this in the document"


# -----------------------------
# 1. Load and split PDF
# -----------------------------


def load_documents():
    pages = PyPDFLoader(str(PDF_PATH)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    documents = []

    for page_number, page in enumerate(pages, start=1):

        chunks = splitter.split_text(page.page_content)

        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"page": page_number, "source": str(PDF_PATH)},
                )
            )

    return documents


# -----------------------------
# 2. Hybrid Search
# -----------------------------


class HybridSearch:

    def __init__(self, documents):
        self.documents = documents
        self.doc_index = {
            (
                doc.page_content.strip(),
                doc.metadata.get("page"),
                doc.metadata.get("source"),
            ): index
            for index, doc in enumerate(documents)
        }

        self.vector_store = self._build_vector_store()

        if not self.vector_store.get_pks(expr="pk >= 0"):
            self.vector_store.add_texts(
                texts=[doc.page_content for doc in documents],
                metadatas=[doc.metadata for doc in documents],
            )

        # Create BM25 index (keyword search remains as-is)
        tokens = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokens)

    def _build_vector_store(self):
        persist_dir = Path(__file__).resolve().parent / "milvus_db"
        persist_dir.mkdir(parents=True, exist_ok=True)

        return Milvus(
            embedding_function=embedding_model,
            connection_args={"uri": str(persist_dir / "employee_handbook_hybrid.db")},
            collection_name="employee_handbook_hybrid",
            index_params={"index_type": "FLAT", "metric_type": "COSINE"},
            auto_id=True,
        )

    def search(self, query):

        # -------------------------
        # Semantic search via Milvus
        # -------------------------
        semantic_results = self.vector_store.similarity_search_with_score(
            query=query, k=20
        )

        vector_scores = {}
        semantic_scores = {}
        for rank, (doc, score) in enumerate(semantic_results):
            key = (
                doc.page_content.strip(),
                doc.metadata.get("page"),
                doc.metadata.get("source"),
            )
            if key in self.doc_index:
                index = self.doc_index[key]
                semantic_scores[index] = score
                vector_scores[index] = vector_scores.get(index, 0) + 1 / (60 + rank + 1)

        # -------------------------
        # BM25 search (no change)
        # -------------------------
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_results = np.argsort(bm25_scores)[::-1][:20]

        # -------------------------
        # Combine results
        # -------------------------
        scores = dict(vector_scores)

        for rank, index in enumerate(bm25_results):
            scores[index] = scores.get(index, 0) + 1 / (60 + rank + 1)

        best = sorted(scores, key=scores.get, reverse=True)[:TOP_K]

        print("\nSemantic search scores (Milvus distance):")
        for index in sorted(semantic_scores, key=semantic_scores.get)[:TOP_K]:
            print(
                f"Page {self.documents[index].metadata.get('page')}: {semantic_scores[index]:.4f}"
            )

        print("\nKeyword search scores (BM25):")
        for index in bm25_results[:TOP_K]:
            print(
                f"Page {self.documents[index].metadata.get('page')}: {bm25_scores[index]:.4f}"
            )

        print("\nHybrid search scores (reciprocal-rank fusion):")
        for index in best:
            print(
                f"Page {self.documents[index].metadata.get('page')}: {scores[index]:.4f}"
            )

        return [self.documents[i] for i in best]


# -----------------------------
# 3. Ask the LLM
# -----------------------------


def answer_question(search_engine, question):

    documents = search_engine.search(question)

    context = "\n\n".join(
        f"[Page {doc.metadata['page']}]\n{doc.page_content}" for doc in documents
    )

    prompt = f"""
Answer the question using ONLY the context below.

If the answer is not present in the context, say:

{NOT_FOUND}

Context:
{context}

Question:
{question}

Answer:
"""

    response = model.invoke(prompt)

    return response.content


# -----------------------------
# 4. Main
# -----------------------------


def main():

    print("Loading PDF...")

    documents = load_documents()

    print(f"Loaded {len(documents)} chunks")

    search_engine = HybridSearch(documents)

    question = input("\nAsk a question: ")

    answer = answer_question(search_engine, question)

    print("\nAnswer:")
    print(answer)


if __name__ == "__main__":
    main()
