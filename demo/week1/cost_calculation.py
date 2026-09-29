
from decimal import Decimal as D

import tiktoken

# ---- Given: fixed pricing table ($ per 1M tokens) ---------------------------
PRICES = {
    "Model A (cheap)":    {"in": D("0.15"), "out": D("0.60")},
    "Model B (frontier)": {"in": D("3.00"), "out": D("15.00")},
}
OUTPUT_TOKENS = 60          # given assumption, not measured
CALLS_PER_MONTH = 50_000
PER_MILLION = D(1_000_000)

# ---- Given: prompts ---------------------------------------------------------
# The line breaks in the original were display wrapping, so the primary version
# joins them with single spaces. Set USE_LITERAL_NEWLINES = True to keep them.
USE_LITERAL_NEWLINES = False

SYSTEM_PROMPT = (
    "You are a support-ticket summarizer. Given a raw ticket, output a one-sentence\n"
    "summary, the customer's sentiment (positive/neutral/negative), and an urgency\n"
    "score from 1-5. Return valid JSON only."
)
USER_INPUT = (
    "Hi, I've been trying to reset my password for the third time today and the\n"
    "reset email never arrives. I checked spam. This is blocking my whole team from\n"
    "accessing the dashboard before our client demo in an hour. Please escalate ASAP,\n"
    "this is extremely frustrating and we're paying for the enterprise plan."
)
if not USE_LITERAL_NEWLINES:
    SYSTEM_PROMPT = SYSTEM_PROMPT.replace("\n", " ")
    USER_INPUT = USER_INPUT.replace("\n", " ")


def call_cost(in_tokens: int, out_tokens: int, price: dict) -> D:
    """cost = (in_tokens * in_price + out_tokens * out_price) / 1,000,000"""
    return (D(in_tokens) * price["in"] + D(out_tokens) * price["out"]) / PER_MILLION


def main() -> None:
    enc = tiktoken.get_encoding("cl100k_base")

    n_sys = len(enc.encode(SYSTEM_PROMPT))
    n_usr = len(enc.encode(USER_INPUT))
    n_in = n_sys + n_usr

    print("== Token counts (cl100k_base) ==")
    print(f"system prompt : {n_sys}")
    print(f"user input    : {n_usr}")
    print(f"total input   : {n_in}")
    print(f"output (given): {OUTPUT_TOKENS}\n")

    costs = {}
    print("== Cost per call and per month ==")
    for name, price in PRICES.items():
        c_in = D(n_in) * price["in"] / PER_MILLION
        c_out = D(OUTPUT_TOKENS) * price["out"] / PER_MILLION
        c_call = c_in + c_out
        c_month = c_call * CALLS_PER_MONTH
        costs[name] = c_call
        print(name)
        print(f"  input : {n_in} x {price['in']} / 1e6           = ${c_in:.8f}")
        print(f"  output: {OUTPUT_TOKENS} x {price['out']} / 1e6            = ${c_out:.8f}")
        print(f"  per call                                 = ${c_call:.8f}")
        print(f"  x {CALLS_PER_MONTH:,} calls/month              = ${c_month:.4f}\n")

    a, b = costs["Model A (cheap)"], costs["Model B (frontier)"]
    monthly_gap = (b - a) * CALLS_PER_MONTH
    print("== Comparison ==")
    print(f"B / A per-call ratio : {b / a:.2f}x")
    print(f"Monthly gap (B - A)  : ${monthly_gap:.4f}\n")

    # ---- Break-even: how many avoided misses per month justify Model B? ----
    # B is worth it if: misses_avoided * cost_per_miss > monthly_gap
    print("== Break-even (illustrative cost of one missed urgent ticket) ==")
    for cost_per_miss in (D(5), D(25), D(100)):
        needed = monthly_gap / cost_per_miss
        print(f"  if a miss costs ${cost_per_miss}: B pays off if it avoids >= {needed:.1f} misses/month")

    # ---- Cascade option: A on everything, B only on a fraction -------------
    escalate_fraction = D("0.20")   # assumption, not given
    n_escalated = int(CALLS_PER_MONTH * escalate_fraction)
    cascade = a * CALLS_PER_MONTH + b * n_escalated
    print(f"\n== Cascade: A on all + B on {escalate_fraction:.0%} ({n_escalated:,} calls) ==")
    print(f"  monthly cost = ${cascade:.4f}")


if __name__ == "__main__":
    main()




# ## The situation

# The model reads a ticket and returns JSON like this:

# ```json
# {"summary": "...", "sentiment": "negative", "urgency": 5}
# ```

# The **urgency score** runs from 1 (can wait) to 5 (drop everything). Support teams use it to decide which tickets get handled first. Your example ticket (team blocked, demo in an hour, enterprise customer) should get a 5.

# ## What is a false negative?

# A false negative is when the model **says "not a problem" but there really was a problem**. Here, the ticket is truly urgent but the model gives it a low score, such as 2. The urgent ticket then sits in the queue and the customer waits.

# There are four possible outcomes, using "urgent" to mean a true urgency of 4 or 5 (my choice for the definition):

# | | Model says urgent (≥4) | Model says not urgent (≤3) |
# |---|---|---|
# | **Ticket truly urgent** | Correct | **False negative**: the costly one |
# | **Ticket not urgent** | False positive: mild annoyance | Correct |

# A false positive just means someone looks at a calm ticket early, which is a small waste. A false negative can lose a customer, so I focus on it. The **false-negative rate** is the number of urgent tickets the model missed divided by the total number of truly urgent tickets. If 50 tickets are truly urgent and the model misses 6, the rate is 6/50 = 12%.

# ## What "switching models" means

# Model A is 23× cheaper but might be less accurate. Model B costs more but might miss fewer urgent tickets. So "when would you switch?" means: at what point is B's extra cost worth paying for the tickets it catches that A misses?

# The extra cost of B is only **$58.45 per month**. That is so small that the deciding question is quality, not price.

# ## How you'd decide, step by step

# 1. **Build a test set.** Take about 500 real tickets and have a human label each one with its true urgency.
# 2. **Run both models** on all 500 and compare their scores to the human labels.
# 3. **Count the false negatives** for each model.
# 4. **Decide.**

# Here is a made-up example (numbers invented for illustration). Say 50 of the 500 are truly urgent.

# - **Case 1:** A misses 6 (12%) and B misses 2 (4%). The gap is 8 points. Scaled to a month, if 5,000 tickets are urgent (my assumption), that is about 400 extra missed tickets. Even at a cost of only $5 per miss, that is $2,000, far more than $58. **Use B.**
# - **Case 2:** A misses 3 (6%) and B misses 3 (6%). There is no gap, so paying $58 more buys nothing. **Use A.**

# ## When to use which

# **Use A when:**
# - Its false-negative rate is about the same as B's (within about 1 percentage point).
# - Its JSON is almost always valid (about 99.5% or better). Broken JSON breaks your pipeline.
# - The task is simple and structured, like this one.

# **Use B when:**
# - A misses noticeably more urgent tickets than B.
# - A often returns invalid JSON.
# - The tickets are hard: long, ambiguous, or sarcastic ("great, another outage, love that for us").

# **Use both (cascade):** run A on every ticket and send only the uncertain ones (say, scored 3 or higher) to B. In the earlier calculation this costs about $14.81/month instead of $61.05.

# ## About my "1 percentage point" number

# That number is a judgment call, not a law. The principle is the break-even formula: B is worth it when the misses it avoids, multiplied by what each miss costs you, exceeds $58.45. Since $58.45 is tiny, a small quality gap is enough to justify B, provided it is a real gap and not noise. With only 50 urgent tickets in a test, a 1–2 point difference could be chance, which is why more labeled urgent tickets give a more trustworthy comparison.

# I can walk through the formula with your own numbers if you'd like. Just tell me how many of your tickets are urgent and what a missed one costs your business.