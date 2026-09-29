from pathlib import Path

from langchain_milvus import Milvus
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedding_model_config import client

pdf_path = Path(__file__).resolve().parent.parent / "books" / "aws_basics.pdf"
persist_dir = Path(__file__).resolve().parent / "milvus_compare"
persist_dir.mkdir(parents=True, exist_ok=True)

loader = PyPDFLoader(str(pdf_path))
docs = loader.load()
print(f"Loaded {len(docs)} pages")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks from the AWS PDF")

query = "Which AWS service is used for storing the images of docker containers?"

configs = [
    ("FLAT", "L2"),
    ("FLAT", "COSINE"),
    ("IVF_FLAT", "L2"),
    ("IVF_FLAT", "COSINE"),
    ("HNSW", "L2"),
    ("HNSW", "COSINE"),
]

final_rows = []

for index_type, metric_type in configs:
    collection_name = f"aws-basics-{index_type.lower()}-{metric_type.lower()}"
    vector_store = Milvus(
        embedding_function=client,
        connection_args={"uri": str(persist_dir / f"{collection_name}.db")},
        collection_name=collection_name,
        index_params={"index_type": index_type, "metric_type": metric_type},
        auto_id=True,
    )

    vector_store.add_documents(chunks)
    results = vector_store.similarity_search_with_score(query=query, k=3)

    print(f"\n=== {index_type} / {metric_type} ===")
    row_entries = []
    for doc, score in results:
        content = doc.page_content.replace("\n", " ")
        print(f"- score={score} | {content[:220]}...")
        row_entries.append({"score": float(score), "content": content[:220]})

    final_rows.append(
        {
            "index_type": index_type,
            "metric_type": metric_type,
            "matches": row_entries,
        }
    )

print("\n=== FINAL COMPARISON SUMMARY ===")
for row in final_rows:
    print(f"\n{row['index_type']} | {row['metric_type']}")
    for match in row["matches"]:
        print(f"  score={match['score']:.6f} | {match['content']}...")
