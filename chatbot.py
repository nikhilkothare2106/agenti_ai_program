from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from model_config import model

chat_history = [SystemMessage(content="You are a helpful assiatant")]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f"AI: {result.content}")
print(chat_history)
