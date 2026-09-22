from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path

loader = PyPDFLoader(str(Path(__file__).parent / "books" / "sample.pdf"))

docs = loader.load() 

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)
