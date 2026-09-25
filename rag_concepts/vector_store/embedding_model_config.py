# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from dotenv import load_dotenv

# load_dotenv()  

# embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")


from langchain_openai import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
)

# texts = [
#     "Phoenix helps with LLM observability.",
#     "Embedding vectors are useful for semantic search.",
# ]

# try:
#     response = client.embeddings.create(
#         model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
#         input=texts,
#     )

#     print(f"Embeddings returned: {len(response.data)}")
#     print(f"Vector size: {len(response.data[0].embedding)}")
#     print("First 8 values of first embedding:", response.data[0].embedding[:8])
# except PermissionDeniedError as e:
#     print("403 PermissionDeniedError from Azure OpenAI.")
#     print("This usually means the Azure OpenAI resource is VNet/private-endpoint restricted.")
#     print("Run this notebook from an allowed network path (for example VPN, VNet VM, or approved subnet).")
#     print("Also verify AZURE_OPENAI_ENDPOINT is the endpoint approved for that network.")
#     print(f"Service message: {e}")