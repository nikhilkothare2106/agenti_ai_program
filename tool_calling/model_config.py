import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)


# import os
# from dotenv import load_dotenv
# from langchain_openai import AzureChatOpenAI
# load_dotenv()

# model = AzureChatOpenAI(
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),   
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),    
#     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),  
#     temperature=0,
# )


# import os
# from dotenv import load_dotenv
# load_dotenv()
# from langchain_google_genai import ChatGoogleGenerativeAI

# model = ChatGoogleGenerativeAI(
#     model= "gemini-3.8-flash",
#     temperature=1.0,
#     max_retries=2,
#     google_api_key=os.getenv("GOOGLE_API_KEY"),
# )
