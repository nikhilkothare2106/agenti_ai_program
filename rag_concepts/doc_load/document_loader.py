from pathlib import Path
from pypdf import PdfReader


def load_txt(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    return {"text": text, "metadata": {"source": filepath, "type": "txt"}}


def load_pdf(filepath: str) -> list[dict]:
    reader = PdfReader(filepath)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""  # can be None for image-only pages
        if text.strip():  # skip blank/scanned pages
            pages.append(
                {
                    "text": text,
                    "metadata": {"source": filepath, "type": "pdf", "page": i + 1},
                }
            )
    return pages


if __name__ == "__main__":
    # sample_path = str(Path(__file__).with_name("sample.txt"))
    # result = load_txt(sample_path)
    # print(result["metadata"])
    # print(result["text"][:200])

    sample_pdf = str(Path(__file__).with_name("sample.pdf"))
    pages = load_pdf(sample_pdf)
    print(f"Loaded {len(pages)} pages from {sample_pdf}")
    for page in pages[:2]:
        print(page["metadata"])
        print(page["text"][:200])
        print("-" * 40)
