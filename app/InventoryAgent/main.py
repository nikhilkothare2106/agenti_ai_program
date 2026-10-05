from collections import OrderedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from bedrock_agentcore.runtime import BedrockAgentCoreApp

from file_tools import tools
from model.load import model as llm
from mcp_client.client import get_streamable_http_mcp_client

# --------------------------------------------------
# OpenTelemetry / LangChain instrumentation
# --------------------------------------------------

LangchainInstrumentor().instrument()


# --------------------------------------------------
# AgentCore application
# --------------------------------------------------

app = BedrockAgentCoreApp()
log = app.logger


# --------------------------------------------------
# System prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are an inventory assistant.

- For inventory tool calls, always use the singular canonical product name.
- Use tools for stock, availability, price, or product lists.
- Never invent inventory data.
- If a user asks about quantity, call check_inventory with the requested quantity.
- After a tool call, answer briefly and clearly.
- Use add_stock to increase stock and remove_stock to decrease stock when asked.
"""


# --------------------------------------------------
# Checkpointer / Memory
# --------------------------------------------------

# Keeps conversation history in memory.
_CHECKPOINT_LIMIT = 128

_checkpointer = InMemorySaver()

# Tracks active session IDs for LRU eviction.
_thread_ids = OrderedDict()


def touch_thread(thread_id: str):
    """
    Keep only the latest 128 active conversation threads.
    """

    if thread_id in _thread_ids:
        _thread_ids.move_to_end(thread_id)
        return

    # Remove least recently used sessions.
    while len(_thread_ids) >= _CHECKPOINT_LIMIT:
        evicted, _ = _thread_ids.popitem(last=False)

        # Delete the corresponding checkpoint/history.
        _checkpointer.delete_thread(evicted)

    _thread_ids[thread_id] = True


# --------------------------------------------------
# Convert message content to plain text
# --------------------------------------------------


def text_of(message) -> str:
    """
    Return plain text regardless of whether message.content
    is a string or structured content blocks.
    """

    content = message.content

    if isinstance(content, str):
        return content

    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


# --------------------------------------------------
# AgentCore entrypoint
# --------------------------------------------------


@app.entrypoint
async def ask_inventory_assistant(payload, context=None):

    log.info("Invoking Inventory Agent...")

    # --------------------------------------------------
    # Get user prompt
    # --------------------------------------------------

    user_input = payload.get("prompt", "")

    if not isinstance(user_input, str):
        raise ValueError("prompt must be a string")

    if not user_input:
        return {"result": 'Please send a JSON body like {"prompt": "..."}.'}

    # --------------------------------------------------
    # Get Runtime session ID
    # --------------------------------------------------

    session_id = getattr(context, "session_id", None) or "default"

    # Keep session in our LRU tracker.
    touch_thread(session_id)

    log.info(f"Session ID: {session_id}")
    log.info(f"Agent input: {user_input}")

    # --------------------------------------------------
    # Load MCP tools
    # --------------------------------------------------

    mcp_client = get_streamable_http_mcp_client()

    mcp_tools = []

    if mcp_client:
        mcp_tools = await mcp_client.get_tools()

    # --------------------------------------------------
    # Combine local + MCP tools
    # --------------------------------------------------

    all_tools = tools + mcp_tools

    # --------------------------------------------------
    # Create LangChain agent
    # --------------------------------------------------

    agent = create_agent(
        model=llm,
        tools=all_tools,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=_checkpointer,
    )

    # --------------------------------------------------
    # Configure conversation thread
    # --------------------------------------------------

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": 12,
    }

    # --------------------------------------------------
    # Invoke agent
    # --------------------------------------------------

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )

    # --------------------------------------------------
    # Get final response
    # --------------------------------------------------

    messages = result.get("messages", [])

    if not messages:
        return {"result": "No response received from the inventory assistant."}

    output = text_of(messages[-1])

    log.info(f"Agent output: {output}")

    return {"result": output}


# --------------------------------------------------
# Start AgentCore locally
# --------------------------------------------------

if __name__ == "__main__":
    app.run()
