import json

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from model_config import model
# ============================================================
# 1. MOCK INVENTORY DATABASE
# ===========================================ma=================

inventory = {
    "iphone 15": {
        "stock": 12,
        "price": 699
    },
    "iphone 15 pro": {
        "stock": 5,
        "price": 999
    },
    "macbook air": {
        "stock": 8,
        "price": 1099
    },
    "airpods pro": {
        "stock": 20,
        "price": 249
    }
}


# ============================================================
# 2. PYTHON FUNCTIONS / TOOLS
# ============================================================

@tool
def check_inventory(product: str, quantity: int = 1) -> str:
    """
    Check whether a product has enough inventory.
    """

    product = product.lower()

    if product not in inventory:
        return json.dumps({
            "success": False,
            "message": f"{product} is not available in inventory"
        })

    available = inventory[product]["stock"]

    return json.dumps({
        "success": True,
        "product": product,
        "requested_quantity": quantity,
        "available_quantity": available,
        "in_stock": available >= quantity
    })


@tool
def get_product_price(product: str) -> str:
    """
    Get the price of a product.
    """

    product = product.lower()

    if product not in inventory:
        return json.dumps({
            "success": False,
            "message": f"{product} is not available"
        })

    return json.dumps({
        "success": True,
        "product": product,
        "price": inventory[product]["price"]
    })


@tool
def list_products() -> str:
    """
    Return all products available in inventory.
    """

    return json.dumps({
        "success": True,
        "products": list(inventory.keys())
    })


tools = [
    check_inventory,
    get_product_price,
    list_products
]


# ============================================================
# 3. LLM
# ============================================================

llm = model

# Bind Python functions to the LLM
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 4. SYSTEM INSTRUCTIONS
# ============================================================

system_prompt = """
You are an inventory assistant.

Your job is to understand the user's natural language request
and use the available tools when inventory information is required.

Rules:

1. Never invent inventory information.
2. Always use a tool when the user asks about:
   - stock
   - availability
   - price
   - available products
3. If the user asks whether a specific quantity is available,
   pass the requested quantity to check_inventory.
4. After receiving the tool result, explain the result clearly.
5. Keep responses concise.

Examples:

User:
"Do you have 5 iPhone 15s?"

Action:
check_inventory(product="iphone 15", quantity=5)

User:
"How much does the MacBook Air cost?"

Action:
get_product_price(product="macbook air")

User:
"What products do you have?"

Action:
list_products()
"""


# ============================================================
# 5. FEW-SHOT EXAMPLES
# ============================================================

few_shot_messages = [
    HumanMessage(
        content="Do you have 3 AirPods Pro?"
    ),

    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "check_inventory",
                "args": {
                    "product": "airpods pro",
                    "quantity": 3
                },
                "id": "example_call_1"
            }
        ]
    ),

    ToolMessage(
        content=json.dumps({
            "success": True,
            "product": "airpods pro",
            "requested_quantity": 3,
            "available_quantity": 20,
            "in_stock": True
        }),
        tool_call_id="example_call_1"
    ),

    AIMessage(
        content="Yes, we have 20 AirPods Pro in stock, so 3 are available."
    )
]


# ============================================================
# 6. FUNCTION CALLING LOOP
# ============================================================

def ask_inventory_assistant(user_input: str):

    messages = [
        SystemMessage(content=system_prompt),

        # Few-shot examples
        *few_shot_messages,

        # Actual user request
        HumanMessage(content=user_input)
    ]

    print("\nUSER:")
    print(user_input)

    # --------------------------------------------------------
    # First LLM call
    # --------------------------------------------------------

    response = llm_with_tools.invoke(messages)

    print("\nLLM RESPONSE:")

    if response.tool_calls:
        print(json.dumps(response.tool_calls, indent=2))
    else:
        print(response.content)

    messages.append(response)

    # --------------------------------------------------------
    # Execute requested tools
    # --------------------------------------------------------

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print("\nFUNCTION CALL:")
        print(f"Function: {tool_name}")
        print(f"Arguments: {tool_args}")

        # Find the corresponding Python function
        selected_tool = {
            "check_inventory": check_inventory,
            "get_product_price": get_product_price,
            "list_products": list_products
        }[tool_name]

        # Execute Python function
        tool_result = selected_tool.invoke(tool_args)

        print("\nFUNCTION RESULT:")
        print(tool_result)

        # Give result back to LLM
        messages.append(
            ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"]
            )
        )

    # --------------------------------------------------------
    # Second LLM call
    # --------------------------------------------------------

    final_response = llm_with_tools.invoke(messages)

    print("\nFINAL ANSWER:")
    print(final_response.content)

    return final_response.content


# ============================================================
# 7. TEST
# ============================================================

if __name__ == "__main__":

    ask_inventory_assistant(
        "I need 5 iPhone 15 phones. Do you have enough?"
    )

    ask_inventory_assistant(
        "What is the price of the MacBook Air?"
    )

    ask_inventory_assistant(
        "Show me all products you currently have."
    )