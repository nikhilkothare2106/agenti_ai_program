import os

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()


def create_model_client(model=None):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set.")

    return OpenAIChatCompletionClient(
        model=model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
            "structured_output": False,
        },
        api_key=api_key,
        base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        temperature=0,
    )


# import os
# from dotenv import load_dotenv
# from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

# load_dotenv()

# def create_model_client(**options):
#     return AzureOpenAIChatCompletionClient(
#         model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#         azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#         api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         temperature=0,
#         **options,
#     )
