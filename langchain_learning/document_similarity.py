from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
import os

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")

embedding = OpenAIEmbeddings(
    model='text-embedding-3-small',
    dimensions=300,
    base_url=base_url,
    api_key=api_key
)


documents = [
    "Sachin Tendulkar is known as the Master Blaster and is one of the greatest batsmen in cricket history.",
    "Virat Kohli is famous for his aggressive batting style and has captained the Indian cricket team in all formats.",
    "MS Dhoni is known for his calm leadership and led India to victory in the 2007 T20 World Cup and 2011 ODI World Cup.",
    "Rohit Sharma is called the Hitman of Indian cricket and holds the record for the highest individual score in an ODI."
]

query = "Tell me about Rohit sharma"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding],doc_embeddings)[0]
print(scores)

# print(list(enumerate(scores)))
index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]
print(score)

print(documents[index])