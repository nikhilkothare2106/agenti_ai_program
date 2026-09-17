import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

model = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=api_key,
    temperature=0,
)


# import os

# from dotenv import load_dotenv

# from langchain_openai import AzureChatOpenAI

# load_dotenv()

# model = AzureChatOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),   
#     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),    
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     temperature=0,
# )