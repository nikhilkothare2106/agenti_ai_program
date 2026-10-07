from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
model = ChatOpenAI(model='openrouter/free',
                   api_key=api_key,
                   base_url=base_url,
                   max_completion_tokens=100)

chat_history = [
    SystemMessage(content='You are a helpful assiatant')
]

while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f'AI: {result.content}')
print(chat_history)