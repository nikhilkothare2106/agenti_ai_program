import json
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langchain.agents import create_agent
from callbacks import LoggingHandler
from file_tools import tools
from model_config import model as llm

MAX_TURNS = 5
MAX_STEPS = 2 * MAX_TURNS + 1
FALLBACK_RESPONSE = "I couldn't complete that request. Please try again or rephrase it."

# This is the agent instruction that tells the model how to behave when it works with inventory tools.
system_prompt = """
    You are an inventory assistant.
    For inventory tool calls, always use the singular canonical product name.
    Use tools for stock, availability, price, or product lists.
    Never invent inventory data.
    If a user asks about quantity, call check_inventory with the requested quantity.
    After a tool call, answer briefly and clearly.
    Use add_stock to increase stock and remove_stock to decrease stock when asked.
"""

checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=checkpointer,
)


# This helper converts a message payload into plain text so we can print or return a clean response.
def text_of(message) -> str:
    """Return plain text from a message or chunk, whatever shape `content` has."""
    c = message.content
    if isinstance(c, str):
        return c
    # content is a list of blocks
    return "".join(
        b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"
    )


def ask_inventory_assistant(user_input: str) -> str:
    """Send a user query to the inventory agent and return the final response text."""

    # This is the production path: one user question, one agent run, and the final answer from the last message.

    config = {
        "configurable": {"thread_id": "nikhil"},
        "recursion_limit": MAX_STEPS,
        "callbacks": [LoggingHandler()],
    }

    try:
        result = agent.invoke({"messages": [("user", user_input)]}, config=config)
    except GraphRecursionError:
        return FALLBACK_RESPONSE
    except Exception:
        return FALLBACK_RESPONSE

    messages = result.get("messages", [])

    if not messages:
        return FALLBACK_RESPONSE

    final_response = text_of(messages[-1])
    print("\nINVENTORY ASSISTANT RESPONSE:")
    print(final_response)

    # The commented blocks below show alternative streaming/debug approaches for learning and inspection.
    # for chunk in agent.stream(
    #     {"messages": [("user", user_input)]},
    #     config={"recursion_limit": MAX_STEPS},
    #     stream_mode="updates",
    # ):
    #     for node, update in chunk.items():
    #         for msg in update["messages"]:
    #             if node == "model":
    #                 if text_of(msg):
    #                     print(f"\nTHOUGHT / ANSWER: {text_of(msg)}")
    #                 for call in msg.tool_calls:
    #                     print(f"ACTION: {call['name']}({json.dumps(call['args'])})")
    #                 final_text = text_of(msg)
    #             elif node == "tools":
    #                 print(f"OBSERVATION: {msg.content}")

    # return final_text

    # for token, metadata in agent.stream(
    #     {"messages": [("user", user_input)]},
    #     stream_mode="messages",
    # ):
    #     if metadata["langgraph_node"] == "model":
    #         print(text_of(token), end="", flush=True)

    # for mode, data in agent.stream(
    #     {"messages": [("user", user_input)]},
    #     stream_mode=["updates", "messages"],
    # ):
    #     if mode == "messages":
    #         token, meta = data
    #         if meta["langgraph_node"] == "model":
    #             print(text_of(token), end="", flush=True)  # live typing
    #     elif mode == "updates":
    #         for node, update in data.items():
    #             if node == "tools":
    #                 for msg in update["messages"]:
    #                     print(f"\n[tool result] {msg.content}")  # step events


if __name__ == "__main__":
    while True:
        user_input = input("\nEnter an inventory query (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break

        print("\nUSER QUERY:")
        print(user_input)

        ask_inventory_assistant(user_input)
