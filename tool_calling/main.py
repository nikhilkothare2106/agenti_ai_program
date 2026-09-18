import json

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
)

from model_config import model as llm
from tools import tools

# Binding Python functions(tools) to the LLM
llm_with_tools = llm.bind_tools(tools)


system_prompt = """
    You are an inventory and weather assistant.
    For inventory tool calls, always use the singular canonical product name.
    Use tools for stock, availability, price, or product lists.
    Use get_current_weather when the user asks about current weather for a city.
    Never invent inventory data.
    If a user asks about quantity, call check_inventory with the requested quantity.
    After a tool call, answer briefly and clearly.
    Use add_stock to increase stock and remove_stock to decrease stock when asked.
"""


def ask_inventory_assistant(user_input: str):

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]

    print("\nUSER:")
    print(user_input)

    response = llm_with_tools.invoke(messages)

    # print("\nLLM RESPONSE:")
    # if response.tool_calls:
    #     print(json.dumps(response.tool_calls, indent=2))
    # else:
    #     print(response.content)

    # Adding AIMessage to the messages
    messages.append(response)

    # Handling the tool calls
    while response.tool_calls:
        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # print("\nFUNCTION CALL:")
            # print(f"Function: {tool_name}")
            # print(f"Arguments: {tool_args}")

            # Finding the corresponding tool function based on the tool name
            selected_tool = {tool.name: tool for tool in tools}[tool_name]

            # Execute Python function
            tool_result = selected_tool.invoke(tool_args)

            # print("\nFUNCTION RESULT:")
            # print(tool_result)

            # Give result back to LLM
            messages.append(
                ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
            )

        response = llm_with_tools.invoke(messages)
        # if response.tool_calls:
        #     print("\nLLM RESPONSE:")
        #     print(json.dumps(response.tool_calls, indent=2))
        messages.append(response)

    print("\nFINAL ANSWER:")
    print(response.content)

    return response.content


# if __name__ == "__main__":

#     ask_inventory_assistant(
#         "I need 5 laptops. Do you have enough?"
#     )

#     ask_inventory_assistant(
#         "What is the price of the smartphone?"
#     )

#     ask_inventory_assistant(
#         "Show me all products you currently have."
#     )

while True:
    user_input = input("\nEnter your query (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    ask_inventory_assistant(user_input)
