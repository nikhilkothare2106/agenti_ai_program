from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import RetryPolicy, Command

from chat_state import ChatState
from file_tools import tools
from hitl import approval_node
from model_config import model as llm

MAX_TURNS = 5
MAX_STEPS = 2 * MAX_TURNS + 1
MAX_RETRIES = 3
FALLBACK_RESPONSE = "I couldn't complete that request. Please try again or rephrase it."

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


tool_node = ToolNode(tools, handle_tool_errors=True)

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

graph.add_node("approval", approval_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node", tools_condition, {"tools": "approval", "__end__": END}
)
graph.add_edge("tools", "chat_node")
chatbot = graph.compile(checkpointer=checkpointer)

config: RunnableConfig = {
    "configurable": {"thread_id": "inventory_thread"},
    "recursion_limit": MAX_STEPS,
}


def text_of(message) -> str:
    """Return plain text from a message or chunk, whatever shape `content` has."""
    c = message.content
    if isinstance(c, str):
        return c
    # content is a list of blocks
    return "".join(
        b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"
    )


def print_trace(messages: list[BaseMessage]) -> None:
    """Print observable tool decisions and results for the latest user turn."""
    latest_user_message = max(
        index for index, message in enumerate(messages) if message.type == "human"
    )
    print("\nTrace:")
    for index, message in enumerate(
        messages[latest_user_message:], start=latest_user_message
    ):
        if message.type == "human":
            print(f"  user: {message.content}")
        elif getattr(message, "tool_calls", None):
            for tool_call in message.tool_calls:
                print(f"  tool call: {tool_call['name']}({tool_call['args']})")
        elif message.type == "tool":
            print(f"  tool result ({message.name}): {message.content}")
        elif message.type == "ai" and index == len(messages) - 1:
            print(f"  assistant: {message.content}")


def ask_inventory_assistant(user_input: str, debug_trace: bool = True) -> str:
    """Send a user query through the inventory agent and return the final answer."""
    answer_parts = []
    try:

        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]}, config=config
        )

        # Graph paused at approval: ask the human, then resume until it finishes
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            print(f"\n{payload['question']} \n {payload['tool_calls']} \n yes/no?")
            answer = input("> ").strip().lower()
            result = chatbot.invoke(Command(resume=answer), config=config)

        if debug_trace:
            print_trace(result["messages"])

        # for token, metadata in chatbot.stream(
        #     {"messages": [("user", user_input)]},
        #     stream_mode="messages",
        #     config=config,
        # ):
        #     if metadata["langgraph_node"] == "chat_node":
        #         text = text_of(token)
        #         if text:
        #             answer_parts.append(text)
        #             if debug_trace:
        #                 print(text, end="", flush=True)

    except Exception as error:
        if debug_trace:
            print(f"Trace stopped: {type(error).__name__}: {error}")


if __name__ == "__main__":
    while True:
        user_input = input("\nEnter an inventory query (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        ask_inventory_assistant(user_input)
