import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def semantic_chunk(text, threshold=0.5):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    embeddings = model.encode(sentences, normalize_embeddings=True)

    chunks = []
    current = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = embeddings[i] @ embeddings[i - 1]  # cosine similarity

        if similarity >= threshold:
            current.append(sentences[i])
        else:
            chunks.append(" ".join(current))
            current = [sentences[i]]

    chunks.append(" ".join(current))
    return chunks


if __name__ == "__main__":
    pdf_path = Path(__file__).resolve().parents[1] / "books" / "aws_basics.pdf"
    documents = PyPDFLoader(str(pdf_path)).load()
    text = "\n".join(document.page_content for document in documents)

    for i, chunk in enumerate(semantic_chunk(text), 1):
        print(f"Chunk {i}: {chunk}\n")
