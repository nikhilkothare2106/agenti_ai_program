from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from pathlib import Path

books_dir = Path(__file__).parent / "books"
loader = DirectoryLoader(path=str(books_dir), glob="*.pdf", loader_cls=PyPDFLoader)

docs = loader.load()
# docs = loader.lazy_load()

for document in docs:
    print(document.metadata)
