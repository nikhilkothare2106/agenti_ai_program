from pydantic import BaseModel, Field

from model_config import model

class Product(BaseModel):
    name: str = Field(description="Name of the product")
    price: float = Field(description="Price of the product")
    quantity: int = Field(description="Available quantity of the product")


structured_model = model.with_structured_output(Product)


product = structured_model.invoke(
    "Extract the product details from this text: "
    "Wireless Keyboard costs rs 49.99 and there are 25 units available."
)

print(product)
print(product.model_dump_json(indent=2))
