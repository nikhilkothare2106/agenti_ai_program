import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_core.tools import FunctionTool
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.tools.http import HttpTool
from model_config import create_model_client

model_client = create_model_client()

schema = {
    "type": "object",
    "properties": {
        "fact": {"type": "string", "description": "A random cat fact"},
        "length": {"type": "integer", "description": "Length of the cat fact"},
    },
    "required": ["fact", "length"],
}

schema_2 = {
    "type": "object",
    "properties": {"fact": {"type": "string"}, "length": {"type": "integer"}},
    "additionalProperties": True,  # Allow additional fields
}

http_tool = HttpTool(
    name="cat_facts_api",
    description="Fetch random cat facts from the Cat Facts API",
    scheme="https",
    host="catfact.ninja",
    port=443,
    path="/fact",
    method="GET",
    return_type="json",
    json_schema=schema_2,
)


# Define a custom function to reverse a string
def reverse_string(
    text: str,
) -> str:
    """Reverse the given text."""
    return text[::-1]


async def main():
    # Create an assistant with the base64 tool
    assistant = AssistantAgent(
        "cat_fact_agent",
        model_client=model_client,
        tools=[http_tool, reverse_string],
        system_message="You are a helpful assistant that can fetch random cat facts (fdirectly call the tool, no changes/inputs) and reverse strings using Tools.",
    )

    # The assistant can now use the base64 tool to decode the string
    response = await assistant.on_messages(
        [
            TextMessage(
                content="Can you please fetch a cat fact using the tool?.",
                source="user",
            )
        ],
        CancellationToken(),
    )
    print(response.chat_message)


asyncio.run(main())
