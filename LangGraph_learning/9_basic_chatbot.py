from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from model_config import model
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):

    # take user query from state
    messages = state["messages"]

    # send to llm
    response = model.invoke(messages)

    # response store state
    return {"messages": [response]}


checkpointer = MemorySaver()
graph = StateGraph(ChatState)

# add nodes
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
# chatbot
# initial_state: ChatState = {
#     'messages': [HumanMessage(content='What is the capital of india')]
# }

# chatbot.invoke(initial_state)['messages'][-1].content
thread_id = "1"

while True:
    user_messgae = input("Type here: ")
    print("User: ", user_messgae)
    if user_messgae.strip().lower() == "exit":
        break

    config = {"configurable": {"thread_id": thread_id}}
    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_messgae)]}, config=config
    )
    print("AI: ", response["messages"][-1].content)
