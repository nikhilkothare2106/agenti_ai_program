import asyncio
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent

from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from model_config import create_model_client

model_client = create_model_client()


assistant = AssistantAgent(
    name="Assistant",
    description="you are a great assistant",
    model_client=model_client,
    system_message="You are a really helpful assistant who help on the task given.",
)


user_agent = UserProxyAgent(
    name="UserProxy",
    description="A proxy agent that represent a user",
    input_func=input,
)


termination_condition = TextMentionTermination("APPROVE")

team = RoundRobinGroupChat(
    participants=[assistant, user_agent], termination_condition=termination_condition
)

stream = team.run_stream(task="Write a nice 4 line poem about India")


async def main():
    await Console(stream)


if __name__ == "__main__":
    asyncio.run(main())
