import json
import re
from decimal import Decimal
from pathlib import Path

from json_repair import repair_json
from pydantic import BaseModel

from model_config import model


# ---------- Schema (identical to C1) ----------
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

RAW_PATH = Path("raw_malformed_output.txt")


def text_of(resp) -> str:
    """Normalize a LangChain message / string into plain text."""
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


# ---------- Strict validation ----------
# Pydantic's default (lax) mode silently coerces "2" -> 2 and ignores extra
# keys, which would hide exactly the failures we want to detect. So we
# validate in strict mode and check for unknown fields explicitly.
def strict_validate(data) -> Invoice:
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must be an object")

    problems = []
    extra = set(data) - set(Invoice.model_fields)
    if extra:
        problems.append(f"Unexpected extra top-level fields: {sorted(extra)}")
    items = data.get("line_items")
    if isinstance(items, list):
        for i, it in enumerate(items):
            if isinstance(it, dict):
                extra_it = set(it) - set(LineItem.model_fields)
                if extra_it:
                    problems.append(f"line_items[{i}] has extra fields: {sorted(extra_it)}")

    invoice = None
    try:
        invoice = Invoice.model_validate(data, strict=True)
    except ValueError as e:  # pydantic ValidationError subclasses ValueError
        problems.append(str(e))

    if problems:
        raise ValueError("\n".join(problems))
    return invoice


# ---------- C2 step 1: induce and capture a malformed output ----------
BAD_PROMPT = f"""Extract the invoice below as JSON with keys: vendor, invoice_number,
line_items (each with description, quantity, unit_price), subtotal, tax_amount,
total_amount, currency.

This is a formatting-robustness test, so follow these output rules exactly:
1. Start with one friendly sentence of prose before the JSON.
2. Put the JSON in a ```json fenced block.
3. Write every quantity as a quoted string, e.g. "2".
4. Add an extra top-level field "notes" containing a short comment.
5. Put a trailing comma after the last line item, and omit the final closing
   brace of the whole JSON object.
6. After the block, add one sentence of prose.

Invoice:
{INVOICE_TEXT}"""


def capture_malformed_output(max_attempts: int = 5) -> str:
    for attempt in range(1, max_attempts + 1):
        raw = text_of(model.invoke(BAD_PROMPT))
        try:
            Invoice.model_validate_json(raw)  # what a naive consumer would do
        except ValueError as e:
            print(f"[capture] attempt {attempt}: naive validation FAILED as intended:")
            print("          " + str(e).splitlines()[0])
            return raw
        print(f"[capture] attempt {attempt}: output was unexpectedly valid, retrying")
    raise RuntimeError("Could not induce malformed output")


if RAW_PATH.exists():
    raw_output = RAW_PATH.read_text()
    print(f"[capture] reusing saved output from {RAW_PATH} (delete it to recapture)")
else:
    raw_output = capture_malformed_output()
    RAW_PATH.write_text(raw_output)

print("\n===== CAPTURED MALFORMED RAW OUTPUT =====")
print(raw_output)
print("=========================================\n")


# ---------- C2 step 2: repair pipeline ----------
# Technique 1: regex / substring extraction of the JSON block
def extract_json_block(raw: str) -> str:
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in output")
    end = raw.rfind("}")
    candidate = raw[start : end + 1] if end > start else raw[start:]
    # If braces are unbalanced the object was truncated: keep everything from
    # the first "{" and let the lenient repair pass close it.
    if candidate.count("{") != candidate.count("}"):
        candidate = raw[start:]
    return candidate.strip()


# Technique 2: lenient JSON repair, then parse
def lenient_parse(block: str):
    repaired = repair_json(block)  # trailing commas, missing braces, etc.
    return json.loads(repaired)


# Technique 3: corrective re-prompt using the pydantic error
def corrective_reprompt(previous_block: str, error: str) -> str:
    prompt = (
        "Your previous output failed validation because:\n"
        f"{error}\n\n"
        "Return corrected JSON only (no prose, no markdown fences, no extra fields). "
        "quantity must be a JSON integer, unit_price/subtotal/tax_amount/total_amount "
        "must be JSON numbers. Preserve every value exactly as it appears in the "
        "invoice; do not recompute any amount.\n\n"
        f"Schema:\n{json.dumps(Invoice.model_json_schema())}\n\n"
        f"Original invoice:\n{INVOICE_TEXT}\n\n"
        f"Previous output:\n{previous_block}"
    )
    return text_of(model.invoke(prompt))


def money(v: float) -> Decimal:
    return Decimal(str(v)).quantize(Decimal("0.01"))


def run_repair_pipeline(raw: str, max_reprompts: int = 2) -> Invoice:
    # --- Stage 1: regex extraction ---
    block = extract_json_block(raw)
    print("--- Stage 1: regex extraction ---")
    print(block)
    try:
        json.loads(block)
        print("=> block is directly parseable JSON\n")
    except json.JSONDecodeError as e:
        print(f"=> block extracted, but still not valid JSON ({e.msg}); needs Stage 2\n")

    # --- Stage 2: lenient repair + strict validation ---
    print("--- Stage 2: json_repair + Invoice validation ---")
    data = lenient_parse(block)
    print(json.dumps(data, indent=2))
    try:
        invoice = strict_validate(data)
        print("=> strict validation PASSED after repair\n")
        return invoice
    except ValueError as e:
        error = str(e)
        print("=> parsed OK, but strict validation FAILED:")
        print(error, "\n")

    # --- Stage 3: corrective re-prompt loop ---
    print("--- Stage 3: corrective re-prompt ---")
    current_block = json.dumps(data)
    for attempt in range(1, max_reprompts + 1):
        reply = corrective_reprompt(current_block, error)
        print(f"[re-prompt {attempt}] model reply:\n{reply}\n")
        try:
            data = lenient_parse(extract_json_block(reply))
            invoice = strict_validate(data)
            print(f"=> strict validation PASSED on re-prompt {attempt}\n")
            return invoice
        except ValueError as e:
            error = str(e)
            current_block = reply
            print(f"=> still failing: {error}\n")
    raise RuntimeError("Repair pipeline exhausted all re-prompts")


repaired = run_repair_pipeline(raw_output)

# ---------- C2 step 3: prove exact recovery ----------
print("===== REPAIRED OBJECT =====")
print(repaired.model_dump_json(indent=2))

Invoice.model_validate(repaired.model_dump(), strict=True)  # zero errors
print("\nInvoice.model_validate(): 0 errors")

exact = repaired == EXPECTED_INVOICE
print(f"Exact match with C1 fixed values: {'PASSED' if exact else 'FAILED'}")
assert exact, f"Mismatch:\n{repaired}\n!=\n{EXPECTED_INVOICE}"

# Non-blocking: the source document itself is internally inconsistent.
li_total = sum((money(i.quantity * i.unit_price) for i in repaired.line_items), Decimal("0"))
print(
    f"\n[document warning] line items sum to {li_total} but stated subtotal is "
    f"{money(repaired.subtotal)}; subtotal + tax == total: "
    f"{money(repaired.subtotal + repaired.tax_amount) == money(repaired.total_amount)}. "
    "This is a defect in the invoice, not in the extraction."
)