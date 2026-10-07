from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model = ChatOpenAI(model='openrouter/free',
                   api_key=api_key,
                   base_url=base_url)

result = model.invoke("What is capital of india?")
print(result.content)