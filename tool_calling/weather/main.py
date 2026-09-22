import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from model_config import model as llm
from weather_tool import tools

llm_with_tools = llm.bind_tools(tools)

system_prompt = """
    You are a weather assistant.
    Use get_current_weather whenever the user asks about current weather for a city.
    After a tool call, answer briefly and clearly.
"""


def ask_weather_assistant(user_input: str) -> str:
    print("\nUSER QUERY:")
    print(user_input)

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    print("\nMODEL TOOL CALLS:")
    print(json.dumps(response.tool_calls, indent=2))

    for tool_call in response.tool_calls:
        selected_tool = {tool.name: tool for tool in tools}[tool_call["name"]]

        tool_result = selected_tool.invoke(tool_call["args"])
        print("\nLOCAL API RESPONSE:")
        print(f"\n{tool_call['name']}: {tool_result}")

        messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))

    response = llm_with_tools.invoke(messages)
    print("\nFINAL ANSWER:")
    print(response.content)
    return response.content


if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a weather query (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        ask_weather_assistant(user_input)
