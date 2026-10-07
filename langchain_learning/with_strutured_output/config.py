from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
# model = ChatOpenAI(model="openrouter/free", api_key=api_key, base_url=base_url)
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)
