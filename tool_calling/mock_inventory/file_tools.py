import json
from pathlib import Path

from langchain_core.tools import tool

INVENTORY_FILE = Path(__file__).with_name("inventory.json")


def read_inventory() -> dict:
    with INVENTORY_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_inventory(inventory: dict) -> None:
    with INVENTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(inventory, file, indent=4)
        file.write("\n")


@tool
def check_inventory(product: str, quantity: int = 1) -> str:
    """Check inventory using a singular product name."""

    product = product.lower()
    inventory = read_inventory()

    if product not in inventory:
        return json.dumps(
            {"success": False, "message": f"{product} is not available in inventory"}
        )

    available = inventory[product]["stock"]
    return json.dumps(
        {
            "success": True,
            "product": product,
            "requested_quantity": quantity,
            "available_quantity": available,
            "in_stock": available >= quantity,
        }
    )


@tool
def get_product_price(product: str) -> str:
    """Get the price using a singular product name."""

    product = product.lower()
    inventory = read_inventory()

    if product not in inventory:
        return json.dumps({"success": False, "message": f"{product} is not available"})

    return json.dumps(
        {"success": True, "product": product, "price": inventory[product]["price"]}
    )


@tool
def list_products() -> str:
    """Return all products available in inventory and their quantities."""

    inventory = read_inventory()
    return json.dumps(
        {
            "success": True,
            "products": [
                {"product": product, "quantity": details["stock"]}
                for product, details in inventory.items()
            ],
        }
    )


@tool
def add_stock(product: str, quantity: int) -> str:
    """Add units using a singular product name."""

    product = product.lower()
    inventory = read_inventory()

    if product not in inventory:
        return json.dumps({"success": False, "message": f"{product} is not available"})
    if quantity < 0:
        return json.dumps(
            {"success": False, "message": "quantity must be greater than or equal to 0"}
        )

    inventory[product]["stock"] += quantity
    write_inventory(inventory)
    return json.dumps(
        {
            "success": True,
            "product": product,
            "added_quantity": quantity,
            "new_stock": inventory[product]["stock"],
        }
    )


@tool
def remove_stock(product: str, quantity: int) -> str:
    """Remove units using a singular product name."""

    product = product.lower()
    inventory = read_inventory()

    if product not in inventory:
        return json.dumps({"success": False, "message": f"{product} is not available"})
    if quantity < 0:
        return json.dumps(
            {"success": False, "message": "quantity must be greater than or equal to 0"}
        )
    if inventory[product]["stock"] < quantity:
        return json.dumps(
            {
                "success": False,
                "message": f"not enough stock to remove {quantity} units from {product}",
                "available_stock": inventory[product]["stock"],
            }
        )

    inventory[product]["stock"] -= quantity
    write_inventory(inventory)
    return json.dumps(
        {
            "success": True,
            "product": product,
            "removed_quantity": quantity,
            "new_stock": inventory[product]["stock"],
        }
    )


tools = [check_inventory, get_product_price, list_products, add_stock, remove_stock]
