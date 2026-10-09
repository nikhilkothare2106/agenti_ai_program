from langchain_core.messages import ToolMessage
from langgraph.types import Command, interrupt

from langgraph_learning.mock_inventory.chat_state import ChatState

SENSITIVE_TOOLS = {"add_stock", "remove_stock"}


def approval_node(state: ChatState):
    """Pause for human approval if stock-exchanging tool is requested."""
    last_message = state["messages"][-1]

    sensitive_tool_call = [
        tool_call
        for tool_call in last_message.tool_calls
        if tool_call["name"] in SENSITIVE_TOOLS
    ]
    if not sensitive_tool_call:
        return Command(goto="tools")  # No sensitive tool calls, continue as normal

    decision = interrupt(
        {
            "question": "Following sensitive tool call was requested. Do you approve?",
            "tool_calls":[{"name": tool_call["name"], "args": tool_call["args"]} for tool_call in sensitive_tool_call]
        }
    )
    if decision == "yes":
        return Command(goto="tools")  # Approved, continue to tools

    rejections = [
        ToolMessage(
            content="Action rejected by human reviewer. Do you want to try again?",
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )
        for tool_call in sensitive_tool_call
    ]

    return Command(goto="chat_node", update={"messages": rejections})
