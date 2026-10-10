import os
from dotenv import load_dotenv
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

load_dotenv()




def create_model_client(**options):
    return AzureOpenAIChatCompletionClient(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0,
        **options,
    )
