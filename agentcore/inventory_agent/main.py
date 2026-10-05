from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

from bedrock_agentcore.runtime import BedrockAgentCoreApp


from file_tools import tools
from model_config import model as llm

system_prompt = """
    You are an inventory assistant.
    For inventory tool calls, always use the singular canonical product name.
    Use tools for stock, availability, price, or product lists.
    Never invent inventory data.
    If a user asks about quantity, call check_inventory with the requested quantity.
    After a tool call, answer briefly and clearly.
    Use add_stock to increase stock and remove_stock to decrease stock when asked.
"""

app = BedrockAgentCoreApp()

# Note: in-memory state lives only as long as the Runtime session's microVM.
checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=checkpointer,
)


def text_of(message) -> str:
    """Return plain text from a message or chunk, whatever shape `content` has."""
    c = message.content
    if isinstance(c, str):
        return c
    return "".join(
        b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"
    )


@app.entrypoint
def ask_inventory_assistant(payload, context=None):
    """Runtime calls this with the JSON body, e.g. {"prompt": "stock of apple?"}."""
    user_input = payload.get("prompt", "")
    if not user_input:
        return {"result": 'Please send a JSON body like {"prompt": "..."}.'}

    # Use the Runtime session id so each session gets its own conversation thread.
    session_id = getattr(context, "session_id", None) or "default"
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 12}

    result = agent.invoke({"messages": [("user", user_input)]}, config=config)
    messages = result.get("messages", [])

    if not messages:
        return {"result": "No response received from the inventory assistant."}

    return {"result": text_of(messages[-1])}


if __name__ == "__main__":
    app.run()
