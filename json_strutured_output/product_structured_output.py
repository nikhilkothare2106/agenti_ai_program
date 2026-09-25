import os

from dotenv import load_dotenv
from openai import AzureOpenAI
from pydantic import BaseModel, Field

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
Model = os.getenv("AZURE_OPENAI_DEPLOYMENT")

class Product(BaseModel):
    name: str = Field(description="Name of the product")
    price: float = Field(description="Price of the product")
    quantity: int = Field(description="Available quantity of the product")



response = client.chat.completions.parse(
    model=Model,
    messages=[
        {
            "role": "user",
            "content": "Generate the name, price and quantity of a fictional Indian product.",
        }
    ],
    response_format=Product,
)

# response = client.chat.completions.create(
#     model=Model,
#     messages=[
#         {
#             "role": "user",
#             "content": "Generate the name, price and quantity of a fictional Indian product in json format.",
#         }
#     ],
#     response_format={
#         "type": "json_object"},
# )

product = response.choices[0].message.parsed
# product = response.choices[0].message.content

print(product)
print(product.model_dump_json(indent=2))
