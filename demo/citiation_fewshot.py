import json
import re
import sys

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from model_config import model  # assumed to be a valid model name string
except ImportError:
    model = None  # --selftest still works without it

REGEX = re.compile(r"^\[[A-Z]+\|\d{4}\|[\w\s]+\]$")

TESTS = {
    "T1": "Yoshua Bengio, Aaron Courville, and Pascal Vincent, 'Representation Learning: "
    "A Review and New Perspectives,' IEEE TPAMI, 2013.",
    "T2": "Vaswani et al., 'Attention Is All You Need,' NeurIPS, 2017.",
    "T3": "Kate Crawford, 'Atlas of AI: Power, Politics, and the Planetary Costs of "
    "Artificial Intelligence,' Yale University Press, 2021.",
}
GOLD = {
    "T1": ("BENGIO", "2013", "Representation Learning A Review and"),
    "T2": ("VASWANI", "2017", "Attention Is All You Need"),
    "T3": ("CRAWFORD", "2021", "Atlas of AI Power Politics"),
}

BASE = (
    "Convert the citation into this exact format: an opening square bracket, the first "
    "author's last name in UPPERCASE letters, a vertical bar, the four-digit publication "
    "year, a vertical bar, the first five words of the title, and a closing square bracket. "
    "Output only the formatted string and nothing else."
)
FEWSHOT = [
    (
        "Yoshua Bengio, Aaron Courville, and Pascal Vincent, 'Representation Learning: "
        "A Review and New Perspectives,' IEEE TPAMI, 2013.",
        "[BENGIO|2013|Representation Learning A Review and]",
    ),
    (
        "Vaswani et al., 'Attention Is All You Need,' NeurIPS, 2017.",
        "[VASWANI|2017|Attention Is All You Need]",
    ),
    (
        "Kate Crawford, 'Atlas of AI: Power, Politics, and the Planetary Costs of "
        "Artificial Intelligence,' Yale University Press, 2021.",
        "[CRAWFORD|2021|Atlas of AI Power Politics]",
    ),
]


def build_prompt(instructions, examples, citation):
    parts = [instructions]
    parts.extend(f"Input: {inp}\nOutput: {out}" for inp, out in examples)
    parts.append(f"Input: {citation}\nOutput:")
    return "\n\n".join(parts)


ARMS = {
    "zero-shot": (BASE, []),
    "few-shot": (BASE, FEWSHOT),
}

# ------------------------------------------------------------------- scoring
BRACKET = re.compile(r"\[([^|\]]*)\|([^|\]]*)\|([^\]]*)\]")


def score(test_id, output):
    """regex_ok = format compliance; facts_ok = right author/year/first-5 words (punctuation-insensitive)."""
    out = output.strip()
    regex_ok = REGEX.fullmatch(out) is not None

    author_g, year_g, title_g = GOLD[test_id]
    m = BRACKET.search(out)
    facts = {"author": False, "year": False, "title": False}
    if m:
        a, y, t = (g.strip() for g in m.groups())
        t = " ".join(re.sub(r"[^\w\s]", "", t).split())
        facts = {
            "author": a.casefold() == author_g.casefold(),
            "year": y == year_g,
            "title": t.casefold() == title_g.casefold(),
        }
    facts_ok = all(facts.values())
    return {
        "output": out,
        "regex_ok": regex_ok,
        "facts_ok": facts_ok,
        "facts": facts,
        "passed": regex_ok and facts_ok,
    }


def report(results):
    """Ask the model to summarize the scored output in plain English."""
    if model is None:
        sys.exit("Could not import `model` from model_config.py.")

    payload = json.dumps(results, ensure_ascii=False, indent=2)
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a strict evaluation assistant. Given the scored results JSON, "
                    "produce a concise human-readable summary with totals and a verdict. "
                    "Keep it short and factual. Use only the actual values in the JSON."
                )
            ),
            HumanMessage(
                content=(
                    "Summarize this evaluation. Include per-arm totals, overall pass counts, "
                    "and a comparison between zero-shot and few-shot.\n\n"
                    f"{payload}"
                )
            ),
        ]
    )
    print(response.content)


def make_generate():
    if model is None:
        sys.exit("Could not import `model` from model_config.py.")

    def generate(prompt, temperature=None):
        if temperature is not None:
            try:
                response = model.with_config(temperature=temperature).invoke(prompt)
            except Exception:
                response = model.invoke(prompt)
        else:
            response = model.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    return generate


# ---------------------------------------------------------------------- main
RUNS = 1
TEMPERATURE = None


def main():
    generate = make_generate()
    arms = dict(ARMS)

    results = {}
    for arm, (instr, examples) in arms.items():
        results[arm] = {}
        for tid, citation in TESTS.items():
            prompt = build_prompt(instr, examples, citation)
            results[arm][tid] = [
                score(tid, generate(prompt, TEMPERATURE)) for _ in range(RUNS)
            ]
    report(results)


if __name__ == "__main__":
    main()
