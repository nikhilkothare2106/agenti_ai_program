# Assignment 1: LLM Foundations — Tokens, Prompts, and Structured Output

**Course:** Week 1 — Fundamentals & Tool Calling
**Format:** Individual work only. No pairs, no groups. Every submission must be one person's own code, prompts, and written answers.
**Time estimate:** 5–7 hours
**Submission:** One PDF/Markdown report + one runnable script or notebook (`assignment1.py` or `assignment1.ipynb`) + a `logs/` folder containing every raw prompt and every raw model output you produced (including failed/wrong attempts)

Every task below gives you the exact input, the exact question, and the exact criteria your output will be checked against. There is no "pick any example you like" — you are working the given data. This is intentional: it's what makes your answer checkable instead of just plausible-sounding.

---

## Learning Objectives

1. Compute, not estimate, the token and dollar cost of a specific LLM request, and reason about context-window budgets under hard constraints.
2. Design a prompt as an interface — system vs. user separation, few-shot as specification — and prove the design choice mattered by testing against a fixed gold answer.
3. Build a structured-output pipeline (schema + validation) against a real messy input, deliberately break it, and implement a repair mechanism that recovers the correct object.

---

## Part A — Tokens, Context Windows, and Cost (30%)

### A1. Exact tokenization (Given / Task / Correct Output)

**Given** — tokenize each of these four strings using the `cl100k_base` tokenizer (via `tiktoken`, `pip install tiktoken`):

```
S1: "The quarterly revenue increased by 12.5% year-over-year."
S2: "こんにちは、今日の天気はどうですか?"
S3: "def calculate_total(items): return sum(i.price * i.qty for i in items)"
S4: "Sub-word tokenization doesn't always split on whitespace—e.g., 'unbelievably' or 'ChatGPT-4o-mini'."
```

**Task:**

1. For each string, report the exact token count and the list of decoded token strings (use `tiktoken`'s `.encode()` then decode each token id individually).
2. For S2 and S4, write 2–3 sentences explaining _why_ the token count is higher or lower per character than S1 (script/encoding effects, subword merges, punctuation handling).

**Correct output criteria:** Your reported token counts must exactly match what `tiktoken.get_encoding("cl100k_base").encode(...)` produces — this is deterministic, so there is a single correct number per string. A mismatch means a bug in your code, not a modeling choice. Include your code output as evidence (not just the number).

---

### A2. Cost calculation under a fixed pricing table (Given / Task / Correct Output)

**Given** — use this fixed pricing table (do **not** look up live prices; these are the numbers you must use so your answer is checkable):

| Model                | Input $ / 1M tokens | Output $ / 1M tokens |
| -------------------- | ------------------- | -------------------- |
| Model A ("cheap")    | $0.15               | $0.60                |
| Model B ("frontier") | $3.00               | $15.00               |

**Given** — this fixed prompt (a support-ticket summarization task). Assume the system prompt below is sent with **every** call:

```
System prompt (fixed, ~40 tokens — count it, don't assume):
"You are a support-ticket summarizer. Given a raw ticket, output a one-sentence
summary, the customer's sentiment (positive/neutral/negative), and an urgency
score from 1-5. Return valid JSON only."

User input (fixed, one example ticket, ~180 tokens — count it, don't assume):
"Hi, I've been trying to reset my password for the third time today and the
reset email never arrives. I checked spam. This is blocking my whole team from
accessing the dashboard before our client demo in an hour. Please escalate ASAP,
this is extremely frustrating and we're paying for the enterprise plan."

Assume expected output length: 60 tokens (fixed assumption — do not measure this
one, use 60).
```

**Task:**

1. Tokenize the system prompt and the user input yourself with `tiktoken` (cl100k_base) to get exact input token counts — do not guess.
2. Compute the exact cost of **one call** for Model A and for Model B.
3. Compute the exact cost of running this same call **50,000 times/month** (realistic support-ticket volume) for both models.
4. State, with a number-backed argument (not just "cheaper is better"), which model you'd deploy in production for this specific use case, and under what condition you'd switch to the other one (e.g., a false-negative urgency-score rate above some threshold).

**Correct output criteria:** Your token counts must match `tiktoken`'s deterministic output (as in A1). Your dollar figures must be arithmetically correct given those token counts and the fixed pricing table above — this is a closed-form calculation with one correct number per cell, so show the formula and the substituted numbers, not just a final figure.

---

## Part B — Prompting as Interface Design (35%)

### B1. System/user separation on a fixed transcript (Given / Task / Correct Output)

**Given** — this exact meeting transcript:

```
"Okay so let's recap. Sarah, can you get the API rate-limiting fix out by
Thursday? Also I think we still need someone to update the pricing page —
Jon, that's you, no rush but let's say end of next week. Oh and I almost
forgot, we need to send the security audit doc to the client, that one's
actually urgent, can someone do that today. I'll handle the follow-up email
to the vendor myself, probably tomorrow."
```

**Gold answer (the correct extraction — used only for grading, not given to you in advance to copy):**

- 4 action items exist in this transcript.
- Each has an assignee, a task description, and a deadline (explicit or relative).
- One item ("security audit doc") has no explicitly named assignee — a correct system prompt design should force the model to either say "unassigned" or infer "the speaker" reasonably, not silently drop it.

**Task:**

1. Write **Version 1**: a single user-message prompt (no system prompt) that asks the model to extract action items from the transcript above, in JSON: `{assignee, task, deadline}[]`.
2. Write **Version 2**: a proper system prompt (defining role, exact output schema, and explicit handling instructions for missing/ambiguous fields) + a minimal user message containing only the transcript.
3. Run both 3 times each (or with temperature > 0 if available) against the exact transcript above.

**Correct output criteria:** Score each run against the gold answer: did it find all 4 items? Did it handle the unassigned item explicitly rather than dropping it or hallucinating a name? Was the JSON schema consistent across all 3 runs? Report a simple scorecard (found 4/4, schema-consistent 3/3, etc.) for both versions and show that Version 2 is measurably more consistent — if it isn't, you must explain why and iterate on your system prompt until it is, since "the interface design didn't help" is not an acceptable conclusion for this task.

---

### B2. Few-shot as specification, against a fixed regex (Given / Task / Correct Output)

**Given** — a citation reformatting task. Convert any input citation into this exact custom format, which is not a standard citation style so the model cannot know it in advance:

```
Target format: [AUTHOR_LASTNAME|YEAR|TRUNCATED_TITLE_5_WORDS]
Example (do not use as your few-shot example — this is the spec, not the prompt):
Input:  "Jennifer Doudna and Emmanuelle Charpentier, 'A Programmable
         Dual-RNA-Guided DNA Endonuclease in Adaptive Bacterial Immunity,'
         Science, 2012."
Output: [DOUDNA|2012|A Programmable Dual-RNA-Guided DNA]
```

**Given** — these 3 fixed test citations to run your prompts against:

```
T1: "Yoshua Bengio, Aaron Courville, and Pascal Vincent, 'Representation
     Learning: A Review and New Perspectives,' IEEE TPAMI, 2013."
T2: "Vaswani et al., 'Attention Is All You Need,' NeurIPS, 2017."
T3: "Kate Crawford, 'Atlas of AI: Power, Politics, and the Planetary Costs
     of Artificial Intelligence,' Yale University Press, 2021."
```

**Task:**

1. Write a **zero-shot** prompt (instructions only, describing the format in words) and run it on T1, T2, T3.
2. Write a **few-shot** prompt (2–3 of your own worked examples, different from T1–T3) and run it on the same T1, T2, T3.

**Correct output criteria:** A correct output for each test case matches this exact regex:

```
^\[[A-Z]+\|\d{4}\|[\w\s]+\]$
```

...**and** the first-author last name, year, and first 5 title words must be factually correct for that citation (e.g., T2's correct output is `[VASWANI|2017|Attention Is All You Need]`). Report a pass/fail per test case per prompt version (6 checks total) and state clearly whether few-shot improved regex compliance, factual correctness, both, or neither — with the actual outputs as evidence.

---

## Part C — Structured Output (35%)

### C1. Schema-constrained extraction from a fixed messy input (Given / Task / Correct Output)

**Given** — this exact schema (implement literally, do not change field names or types):

```python
from pydantic import BaseModel

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
```

**Given** — this exact messy invoice text as input:

```
INVOICE
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
```

**Task:**

1. Use structured-output generation (JSON mode / function-calling with schema / `instructor` — your choice, name which one you used) to extract this invoice into the `Invoice` model above.
2. Validate the result against the schema and against the arithmetic in the invoice (does `subtotal + tax_amount == total_amount`? do the line items' `quantity × unit_price` sum to the stated subtotal?).

**Correct output criteria:** The only correct extraction has `vendor="Blue Ridge Office Supplies LLC"`, `invoice_number="BR-2024-00931"`, 3 line items with exactly the quantities/prices shown, `subtotal=716.48`, `tax_amount=53.74`, `total_amount=770.22`, `currency="USD"`. There is one correct object here — this is not a matter of style. If your model's extraction disagrees with any of these fixed values, that is a failure to fix, not a valid alternative interpretation.

### C2. Deliberately induced failure, then repair (Given / Task / Correct Output)

**Task:**

1. Construct a variant of the C1 prompt or input that reliably produces **invalid** output — at least one of: truncated/unparseable JSON, prose wrapped around the JSON, a hallucinated extra field, or a wrong type (e.g., `quantity` returned as a string). You must show the actual malformed raw output you captured, not a hypothetical.
2. Implement a repair pipeline using **at least two** of the following, and show each one running successfully on your captured malformed output:
    - Regex/substring extraction of the JSON block from surrounding prose
    - A lenient JSON repair pass (e.g., `json_repair` or hand-written) followed by `Invoice.model_validate()`
    - A corrective re-prompt ("your previous output failed validation because: `<pydantic error>` — return corrected JSON only") and a second model call
3. Prove your repair pipeline recovers the **exact correct object from C1** (same field values) from the malformed input.

**Correct output criteria:** The repaired object must pass `Invoice.model_validate()` with zero errors and must match the fixed correct values from C1 exactly. A repair that "produces valid JSON but with wrong numbers" does not count as success — validation passing is necessary but not sufficient.

---

## Submission Checklist

- [ ] A1: token counts + decoded tokens for S1–S4, with explanation
- [ ] A2: exact token counts for system+user prompt, cost table for Model A/B at 1 call and 50,000 calls, model choice justified
- [ ] B1: both prompt versions, 3 runs each, scorecard vs. gold answer
- [ ] B2: both prompt versions, outputs for T1–T3, 6-cell pass/fail table
- [ ] C1: schema code, extraction matching the fixed correct values exactly
- [ ] C2: captured malformed output, repair code, proof it recovers the exact C1 values
- [ ] `logs/` folder with every raw prompt and raw output, including failed attempts
- [ ] Confirmation this is entirely your own individual work
