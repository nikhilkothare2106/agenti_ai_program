from typing import Annotated, TypedDict
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy

from file_tools import tools
from model_config import model as llm

MAX_STEPS = 10
MAX_RETRIES = 3

checkpointer = InMemorySaver()

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """
    You are an inventory assistant.
    For inventory tool calls, always use the singular canonical product name.
    Use tools for stock, availability, price, or product lists.
    Never invent inventory data.
    If a user asks about quantity, call check_inventory with the requested quantity.
    After a tool call, answer briefly and clearly.
    Use add_stock to increase stock and remove_stock to decrease stock when asked.
""".strip()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    recent_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=len,
        max_tokens=30,
        start_on="human",
        end_on=("human", "tool"),
        include_system=False,
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT), *recent_messages]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Tool exceptions go back to the model as a ToolMessage instead of crashing the run
tool_node = ToolNode(tools)

graph = StateGraph(ChatState)
graph.add_node(
    "chat_node",
    chat_node,
    retry_policy=RetryPolicy(
        max_attempts=MAX_RETRIES,
        initial_interval=1.0,
        backoff_factor=2.0,
    ),
)
graph.add_node(
    "tools",
    tool_node,
    retry_policy=RetryPolicy(
        max_attempts=MAX_RETRIES,
        initial_interval=0.1,
        backoff_factor=2.0,
    ),
)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
chatbot = graph.compile(checkpointer=checkpointer)

config: RunnableConfig = {
    "configurable": {"thread_id": "inventory_thread"},
    "recursion_limit": MAX_STEPS,
}


def ask_inventory_assistant(user_input: str) -> str:
    """Send a user query through the inventory agent and return the final answer."""
    try:
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )
        answer = result["messages"][-1].content
    except GraphRecursionError:
        answer = "I couldn't complete that request in a reasonable number of steps. Please try rephrasing it."
    except Exception as e:
        # Raised after all LLM retries are exhausted
        answer = f"Something went wrong while processing your request: {e}"
    print(answer)
    return answer


if __name__ == "__main__":
    while True:
        user_input = input("\nEnter an inventory query (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        ask_inventory_assistant(user_input)
