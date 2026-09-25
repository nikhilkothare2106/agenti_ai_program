from decimal import Decimal
from pydantic import BaseModel
from model_config import model


class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float


class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    line_items: list[LineItem]
    subtotal: float
    tax_amount: float
    total_amount: float
    currency: str


INVOICE_TEXT = """INVOICE
From: Blue Ridge Office Supplies LLC
Inv# BR-2024-00931

Item                          Qty    Unit Price
Ergonomic desk chair           2       $189.99
Monitor stand (dual)           3        $45.50
USB-C hub, 7-port               5        $22.00

Subtotal: $716.48
Tax (7.5%): $53.74
TOTAL DUE: $770.22
Currency: USD
"""


EXPECTED_INVOICE = Invoice(
    vendor="Blue Ridge Office Supplies LLC",
    invoice_number="BR-2024-00931",
    line_items=[
        LineItem(description="Ergonomic desk chair", quantity=2, unit_price=189.99),
        LineItem(description="Monitor stand (dual)", quantity=3, unit_price=45.50),
        LineItem(description="USB-C hub, 7-port", quantity=5, unit_price=22.00),
    ],
    subtotal=716.48,
    tax_amount=53.74,
    total_amount=770.22,
    currency="USD",
)


def money(value: float) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.01"))


def validate_invoice(invoice: Invoice) -> None:
    errors = []

    fixed_values_match = invoice == EXPECTED_INVOICE
    print(
        f"Fixed-value extraction check: {'PASSED' if fixed_values_match else 'FAILED'}"
    )
    if not fixed_values_match:
        errors.append(
            "The extracted invoice does not match the required fixed values: "
            f"{invoice.model_dump_json()}"
        )

    line_items_total = sum(
        (money(item.quantity * item.unit_price) for item in invoice.line_items),
        Decimal("0.00"),
    )
    line_items_match = line_items_total == money(invoice.subtotal)
    print(
        f"Line-item subtotal check: {'PASSED' if line_items_match else 'FAILED'} "
        f"({line_items_total} vs {money(invoice.subtotal)})"
    )
    if not line_items_match:
        errors.append(
            f"Line items total {line_items_total} does not equal "
            f"subtotal {money(invoice.subtotal)}"
        )

    totals_match = money(invoice.subtotal + invoice.tax_amount) == money(
        invoice.total_amount
    )
    print(
        f"Subtotal-plus-tax check: {'PASSED' if totals_match else 'FAILED'} "
        f"({money(invoice.subtotal + invoice.tax_amount)} vs {money(invoice.total_amount)})"
    )
    if not totals_match:
        errors.append(
            f"Subtotal plus tax {money(invoice.subtotal + invoice.tax_amount)} does not "
            f"equal total {money(invoice.total_amount)}"
        )

    if errors:
        raise ValueError("Invoice validation failed:\n- " + "\n- ".join(errors))


structured_model = model.with_structured_output(Invoice)

extracted_invoice = structured_model.invoke(
    "Extract the invoice into the supplied Invoice schema. Preserve every value "
    "exactly, including all descriptions, quantities, prices, and amounts.\n\n"
    + INVOICE_TEXT
)
print(extracted_invoice.model_dump_json(indent=2))
validate_invoice(extracted_invoice)
print("Schema and arithmetic validation passed.")
