from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from model_config import embedding_model, model

load_dotenv()

PDF_PATH = Path(__file__).parent.parent / "book" / "employee_handbook.pdf"
DB_PATH = Path(__file__).parent / "milvus_db" / "employee_handbook_hybrid.db"

CANDIDATES = 20  # how many chunks to pass from hybrid search to the reranker
TOP_K = 5        # how many chunks the LLM finally sees
NOT_FOUND = "I could not find this in the document"


# -----------------------------
# 1. Load and split PDF
# -----------------------------


def load_chunks():
    pages = PyPDFLoader(str(PDF_PATH)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    return splitter.split_documents(pages)  # keeps page metadata automatically


# -----------------------------
# 2. Hybrid search + reranking
# -----------------------------


class HybridSearch:
    def __init__(self, chunks):
        self.chunks = chunks

        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store = Milvus(
            embedding_function=embedding_model,
            connection_args={"uri": str(DB_PATH)},
            collection_name="employee_handbook_hybrid",
            index_params={"index_type": "FLAT", "metric_type": "COSINE"},
            auto_id=True,
        )
        if not self.vector_store.get_pks(expr="pk >= 0"):
            self.vector_store.add_documents(chunks)

        self.bm25 = BM25Okapi([c.page_content.lower().split() for c in chunks])
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def hybrid_search(self, query):
        semantic = self.vector_store.similarity_search(query, k=20)
        keyword = self.bm25.get_top_n(query.lower().split(), self.chunks, n=20)

        # Reciprocal-rank fusion: reward chunks ranked high in either list
        scores, docs = {}, {}
        for results in (semantic, keyword):
            for rank, doc in enumerate(results):
                key = doc.page_content
                docs[key] = doc
                scores[key] = scores.get(key, 0) + 1 / (60 + rank + 1)

        best = sorted(scores, key=scores.get, reverse=True)[:CANDIDATES]
        return [docs[key] for key in best]

    def rerank(self, query, docs):
        # The cross-encoder reads the question and each chunk together
        scores = self.reranker.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:TOP_K]]

    def search(self, query):
        candidates = self.hybrid_search(query)
        return self.rerank(query, candidates)

# -----------------------------
# 3. Ask the LLM
# -----------------------------

def answer_question(search_engine, question):
    docs = search_engine.search(question)
    context = "\n\n".join(
        f"[Page {d.metadata['page'] + 1}]\n{d.page_content}" for d in docs
    )
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say: {NOT_FOUND}

Context:
{context}

Question: {question}
Answer:"""
    return model.invoke(prompt).content


# -----------------------------
# 4. Main
# -----------------------------


def main():
    print("Loading PDF...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    search_engine = HybridSearch(chunks)
    question = input("\nAsk a question: ")
    print("\nAnswer:")
    print(answer_question(search_engine, question))


if __name__ == "__main__":
    main()