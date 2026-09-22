import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from file_tools import tools
from model_config import model as llm

llm_with_tools = llm.bind_tools(tools)

system_prompt = """
You are an inventory assistant.
For inventory tool calls, always use the singular canonical product name.
Use tools for stock, availability, price, or product lists.
Never invent inventory data.
If a user asks about quantity, call check_inventory with the requested quantity.
After a tool call, answer briefly and clearly.
Use add_stock to increase stock and remove_stock to decrease stock when asked.
"""


def ask_inventory_assistant(user_input: str) -> str:
    print("\nUSER QUERY:")
    print(user_input)

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    print("\nMODEL TOOL CALLS:")
    print(json.dumps(response.tool_calls, indent=2))


    print("\nLOCAL API RESPONSE:")
    for tool_call in response.tool_calls:
        selected_tool = {tool.name: tool for tool in tools}[tool_call["name"]]

        tool_result = selected_tool.invoke(tool_call["args"])
        print(f"\n{tool_call['name']}: {tool_result}")

        messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

    response = llm_with_tools.invoke(messages)
    print("\nFINAL ANSWER:")
    print(response.content)
    return response.content


if __name__ == "__main__":
    while True:
        user_input = input("\nEnter an inventory query (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        ask_inventory_assistant(user_input)
