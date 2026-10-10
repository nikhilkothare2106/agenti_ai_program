import asyncio
from autogen_agentchat.agents import AssistantAgent

from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from model_config import create_model_client

model_client = create_model_client()


assistant = AssistantAgent(
    name="Writer",
    description="you are a great writer",
    model_client=model_client,
    system_message="You are a really helpful writer who writes in less then 30 words.",
)

assistant2 = AssistantAgent(
    name="Reviewer",
    description="you are a great reviewer",
    model_client=model_client,
    system_message="You are a really helpful reviewer who writes in less then 30 words..",
)

assistant3 = AssistantAgent(
    name="Editor",
    description="you are a great editor",
    model_client=model_client,
    system_message="You are a really helpful editor who writes in less then 30 words..",
)


team = RoundRobinGroupChat(
    participants=[assistant, assistant2, assistant3], max_turns=3
)


async def main():
    task = " Write a 3 line poem about sky"

    while True:
        stream = team.run_stream(task=task)
        await Console(stream)

        feedback = input('Please Provide your feedback (type "exit" to stop)')
        if feedback.lower().strip() == "exit":
            break

        task = feedback


if __name__ == "__main__":
    asyncio.run(main())
