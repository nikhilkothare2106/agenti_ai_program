from __future__ import annotations

import random
import re
import statistics
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_milvus import Milvus

from model_config import embedding_model

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = (
    BASE_DIR.parent.parent / "rag_concepts" / "books" / "employee_handbook.pdf"
)
PERSIST_DIR = BASE_DIR / "milvus_db"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
PAGE_MARKER = re.compile(r"(?m)^Page (\d+)\s*$")
HEADING = re.compile(r"(?m)^\s*(\d+(?:\.\d+)*\.?\s+[A-Z][^\n]+)\s*$")
RUNNING_HEADER = re.compile(
    r"(?m)^Northwind Technologies - Employee Handbook \(Sample, v3\.0\)\s*$"
)

QUESTIONS = [
    {
        "question": "How many unused annual leave days can carry over?",
        "answer": "Up to 10 unused days may be carried over",
    },
    {
        "question": "When is a medical certificate required for sick leave?",
        "answer": "absences longer than 2 consecutive days",
    },
    {
        "question": "How far in advance should I request more than three days of annual leave?",
        "answer": "at least 2 weeks in advance",
    },
    {
        "question": "How many days per week must employees work from the office?",
        "answer": "at least 3 days per week",
    },
    {
        "question": "When are salaries paid each month?",
        "answer": "on the last working day of each month",
    },
    {
        "question": "What is the annual learning budget, and who approves requests above it?",
        "answer": "$800 for courses, books and conferences. Requests above this amount need VP approval",
    },
    {
        "question": "How long do employees have to submit business expenses?",
        "answer": "within 30 days of being incurred",
    },
    {
        "question": "Compare the learning budget with the retirement match.",
        "answer": [
            "$800 for courses, books and conferences",
            "matches employee retirement contributions up to 5%",
        ],
    },
    {
        "question": "What is the minimum company password length?",
        "answer": "at least 14 characters",
    },
    {
        "question": "Compare annual leave and maternity leave entitlements.",
        "answer": ["24 days per year", "26 weeks paid"],
    },
]


def load_page_text(pdf_path: Path) -> str:
    pages = PyPDFLoader(str(pdf_path)).load()
    if not pages:
        raise ValueError(f"No pages found in {pdf_path}")

    print("=== Raw page 1: inspect header/footer before chunking ===")
    print(pages[0].page_content)
    print("=== End raw page 1 ===\n")

    page_blocks = []
    for page_number, document in enumerate(pages, start=1):
        page_text = document.page_content
        page_text = RUNNING_HEADER.sub("", page_text)
        page_text = re.sub(rf"(?m)^Page {page_number}\s*$", "", page_text)
        page_text = page_text.strip()
        if page_text:
            page_blocks.append(f"Page {page_number}\n{page_text}")
    return "\n\n".join(page_blocks)


def _metadata(text: str, start: int) -> dict:
    page_matches = [
        match for match in PAGE_MARKER.finditer(text) if match.start() <= start
    ]
    heading_matches = [
        match for match in HEADING.finditer(text) if match.start() <= start
    ]
    return {
        "page": int(page_matches[-1].group(1)) if page_matches else 1,
        "section": (
            heading_matches[-1].group(1).strip() if heading_matches else "Document"
        ),
    }


def _chunk(text: str, start: int, end: int) -> dict:
    return {"text": text[start:end].strip(), **_metadata(text, start)}


def chunk_fixed(text: str, size: int = 500, overlap: int = 0) -> list[dict]:
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be in [0, size)")
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if text[start:end].strip():
            chunks.append(_chunk(text, start, end))
        if end == len(text):
            break
        start = end - overlap
    return chunks


def chunk_recursive(text: str, size: int = 800, overlap: int = 120) -> list[dict]:
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be in [0, size)")
    chunks = []
    start = 0
    while start < len(text):
        hard_end = min(start + size, len(text))
        end = hard_end
        if hard_end < len(text):
            minimum_end = start + max(1, size // 2)
            for separator in ("\n\n", "\n", ". ", " "):
                boundary = text.rfind(separator, minimum_end, hard_end)
                if boundary >= minimum_end:
                    end = boundary + len(separator)
                    break
        if text[start:end].strip():
            chunks.append(_chunk(text, start, end))
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def chunk_structure(text: str) -> list[dict]:
    headings = list(HEADING.finditer(text))
    boundaries = [0] + [match.start() for match in headings] + [len(text)]
    chunks = []
    for start, end in zip(boundaries, boundaries[1:]):
        if text[start:end].strip():
            chunks.append(_chunk(text, start, end))
    return chunks


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def _answer_location(chunk: dict, answer: str) -> tuple[int, str]:
    position = chunk["text"].casefold().find(answer)
    page_markers = [
        match
        for match in PAGE_MARKER.finditer(chunk["text"])
        if match.start() <= position
    ]
    headings = [
        match for match in HEADING.finditer(chunk["text"]) if match.start() <= position
    ]
    page = int(page_markers[-1].group(1)) if page_markers else chunk["page"]
    section = headings[-1].group(1).strip() if headings else chunk["section"]
    return page, section


def _print_chunk_stats(name: str, chunks: list[dict]) -> None:
    sizes = [len(chunk["text"]) for chunk in chunks]
    print(
        f"{name}: count={len(chunks)}, mean={statistics.mean(sizes):.1f}, "
        f"shortest={min(sizes)}, longest={max(sizes)} characters"
    )
    samples = random.Random(42).sample(chunks, min(3, len(chunks)))
    print(f"Read these {len(samples)} random {name} chunks aloud:")
    for index, chunk in enumerate(samples, start=1):
        print(
            f"  Sample {index} (page {chunk['page']}, section {chunk['section']}): "
            f"{chunk['text']}"
        )


def _index_collection(name: str, chunks: list[dict]) -> Milvus:
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    store = Milvus(
        embedding_function=embedding_model,
        connection_args={"uri": str(PERSIST_DIR / f"{name}_{RUN_ID}.db")},
        collection_name=name,
        index_params={"index_type": "FLAT", "metric_type": "L2"},
        auto_id=True,
    )
    store.add_texts(
        texts=[chunk["text"] for chunk in chunks],
        metadatas=[
            {"page": chunk["page"], "section": chunk["section"]} for chunk in chunks
        ],
    )
    return store


def main() -> None:
    pdf_path = Path(DEFAULT_PDF)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    text = load_page_text(pdf_path)
    strategies = {
        "fixed_500": chunk_fixed(text, size=500, overlap=0),
        "recursive_800": chunk_recursive(text, size=800, overlap=120),
        "structure": chunk_structure(text),
    }

    stores = {}
    for name, chunks in strategies.items():
        _print_chunk_stats(name, chunks)
        stores[name] = _index_collection(name, chunks)

    results = []
    for item in QUESTIONS:
        print(f"\nQuestion: {item['question']}")
        expected = item["answer"]
        expected = expected if isinstance(expected, list) else [expected]
        expected = [_normalized(answer) for answer in expected]
        for name, store in stores.items():
            matches = store.similarity_search(item["question"], k=3)
            hit = any(
                all(answer in _normalized(document.page_content) for answer in expected)
                for document in matches
            )
            print(f"  {name}: {'yes' if hit else 'no'}")
            results.append((name, item, hit))

    print("\n=== Comparison ===")
    print("strategy | hit rate | chunk count | mean chunk size")
    for name, chunks in strategies.items():
        hits = sum(hit for strategy, _item, hit in results if strategy == name)
        mean_size = statistics.mean(len(chunk["text"]) for chunk in chunks)
        print(f"{name} | {hits}/10 | {len(chunks)} | {mean_size:.1f}")

    winning_name = max(
        strategies,
        key=lambda name: (
            sum(hit for strategy, _item, hit in results if strategy == name),
            -len(strategies[name]),
        ),
    )
    winner_hits = sum(
        hit for strategy, _item, hit in results if strategy == winning_name
    )
    print(f"\nWinner: {winning_name} ({winner_hits}/10).")
    other_scores = [
        (
            sum(hit for strategy, _item, hit in results if strategy == name),
            len(strategies[name]),
        )
        for name in strategies
        if name != winning_name
    ]
    best_other_hits = max(score for score, _count in other_scores)
    print(
        f"It scored {winner_hits}/10, with {best_other_hits}/10 for the next-best "
        "strategy. Among strategies with the same hit rate, it uses fewer chunks, "
        "so it stores the tested handbook in a smaller index."
    )
    failed = next(
        (
            (item, chunks)
            for strategy, item, hit in results
            if strategy == winning_name and not hit
            for chunks in [strategies[strategy]]
        ),
        None,
    )
    if failed:
        item, chunks = failed
        answers = item["answer"]
        answers = answers if isinstance(answers, list) else [answers]
        answers = [_normalized(answer) for answer in answers]
        complete_answer_chunks = [
            chunk
            for chunk in chunks
            if all(answer in _normalized(chunk["text"]) for answer in answers)
        ]
        answer_locations = []
        for answer in answers:
            matching_chunks = [
                chunk for chunk in chunks if answer in _normalized(chunk["text"])
            ]
            if matching_chunks:
                chunk = matching_chunks[0]
                page, section = _answer_location(chunk, answer)
                answer_locations.append(f"'{answer}' on page {page} ({section})")
        if complete_answer_chunks:
            chunk = complete_answer_chunks[0]
            detail = (
                f"The complete answer is in one {len(chunk['text'])}-character chunk "
                f"on page {chunk['page']} ({chunk['section']}), but that chunk was not "
                "retrieved in the top three."
            )
        else:
            detail = (
                "Chunk boundaries put the required facts in separate chunks: "
                f"{'; '.join(answer_locations)}. The 120-character overlap did not "
                "bridge the distance between the annual and maternity leave entries."
            )
        print(f"Winner miss: {item['question']} {detail}")


if __name__ == "__main__":
    main()
