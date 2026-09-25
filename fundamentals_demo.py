#!/usr/bin/env python3

import os
import sys
import json
import textwrap

from model_config import model

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import tool

HAVE_KEY = bool(os.environ.get("GROQ_API_KEY") or os.environ.get("API_KEY"))


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def wrap(text, width=68):
    return "\n".join(textwrap.wrap(text, width)) if text else text


def get_chat(temperature=1.0, **kwargs):
    """Return the shared model instance configured in model_config."""
    return model


def build_messages(user_msg, system=None):
    msgs = []
    if system:
        msgs.append(SystemMessage(content=system))
    msgs.append(HumanMessage(content=user_msg))
    return msgs


def call(user_msg, system=None, temperature=1.0, max_tokens=400, **kwargs):
    """Thin wrapper around ChatGroq.invoke(). Returns the AIMessage, or
    None (with an explanation printed) if no API key is set."""
    if not HAVE_KEY:
        print("[no GROQ_API_KEY set -- skipping live call]")
        return None
    chat = get_chat(temperature=temperature, max_tokens=max_tokens, **kwargs)
    return chat.invoke(build_messages(user_msg, system=system))


def print_response(resp):
    if resp is None:
        return
    if isinstance(resp, AIMessage) and resp.tool_calls:
        for tc in resp.tool_calls:
            print(f"[tool_use] {tc['name']}({json.dumps(tc['args'])})")
    if resp.content:
        print(wrap(resp.content))


# ---------------------------------------------------------------------
# 1. TOKENIZATION
# ---------------------------------------------------------------------
def demo_tokenization():
    banner("1. TOKENIZATION")
    sample = (
        "LLMs don't read words -- they read tokens, which can be "
        "whole words, sub-words, or punctuation."
    )
    print("Sample text:")
    print(wrap(sample))

    if HAVE_KEY:
        try:
            chat = get_chat()
            approx = chat.get_num_tokens(sample)
            print(
                f"\nApproximate token count (tiktoken-based estimate "
                f"via LangChain): {approx}"
            )
            print(
                "(Not the exact Llama tokenizer -- Groq doesn't expose "
                "one via the API. Treat this as a ballpark figure.)"
            )
        except Exception as e:
            print(f"\n(get_num_tokens failed: {e})")
    else:
        approx = max(1, len(sample) // 4)
        print(f"\nNo API key -- rough heuristic estimate: ~{approx} tokens")
        print("(rule of thumb: ~4 characters, or ~0.75 words, per token)")

    print(
        wrap(
            "\nWhy it matters: you're billed and rate-limited per token, not "
            "per character or word. Prompt length, context window limits, and "
            "cost all come back to token counts. Different model families "
            "(Claude, Llama, GPT) use different tokenizers, so the same text "
            "can produce different token counts across providers."
        )
    )


# ---------------------------------------------------------------------
# 2. CONTEXT WINDOW
# ---------------------------------------------------------------------
def demo_context_window():
    banner("2. CONTEXT WINDOW")
    print(
        wrap(
            "The context window is the maximum number of tokens (input + "
            "output combined) a model can 'see' in one request. Groq-hosted "
            "models each have their own limit set by the underlying model "
            "(e.g. Llama 3.3 70B currently supports a large context window) "
            "-- check https://console.groq.com/docs/models for the exact "
            "current figure per model, since it varies and changes over time."
        )
    )
    print(
        wrap(
            "\nPractical implications:\n"
            "  - Long documents or chat histories must be trimmed, summarized, "
            "or chunked once they approach the limit.\n"
            "  - Total input+output tokens must fit inside the window -- "
            "max_tokens (output) competes with everything else you've sent.\n"
            "  - Retrieval-augmented generation (RAG) exists largely to avoid "
            "stuffing everything into context.\n"
            "  - Groq's differentiator isn't a bigger window -- it's raw "
            "inference speed (their LPU hardware), so the same context-window "
            "tradeoffs apply, just resolved faster per request."
        )
    )


# ---------------------------------------------------------------------
# 3. TEMPERATURE
# ---------------------------------------------------------------------
def demo_temperature():
    banner("3. TEMPERATURE")
    print(
        wrap(
            "Temperature controls sampling randomness. 0 = near-deterministic, "
            "always picks the highest-probability next token. Higher values "
            "(e.g. 1.0) flatten the probability distribution, letting less "
            "likely tokens get picked -- more variety, more risk of going "
            "off the rails. ChatGroq exposes this as the `temperature` "
            "constructor argument, same shape as most chat model APIs."
        )
    )
    prompt = "In exactly one sentence, describe a rainy city street."
    for temp in (0.0, 1.0):
        print(f"\n--- temperature={temp} ---")
        resp = call(prompt, temperature=temp, max_tokens=60)
        if resp:
            print_response(resp)
        else:
            canned = {
                0.0: "Rain slicked the empty street, streetlights blurring "
                "into long amber reflections.",
                1.0: "Neon puddles shimmered under a hiss of rain as taxis "
                "sighed past shuttered noodle stalls.",
            }[temp]
            print("[canned example] " + canned)
    print(
        wrap(
            "\nRun the low-temperature call twice and you'll get nearly "
            "identical output every time. Run the high-temperature call twice "
            "and wording will vary noticeably -- same idea, different phrasing."
        )
    )


# ---------------------------------------------------------------------
# 4. BASE VS. INSTRUCT MODELS
# ---------------------------------------------------------------------
def demo_base_vs_instruct():
    banner("4. BASE VS. INSTRUCT MODELS")
    print(
        wrap(
            "A BASE model is trained purely to predict the next token from "
            "raw text (books, web pages, code). Given a prompt, it tends to "
            "*continue* the text rather than *answer* it -- it has no notion "
            "of 'the user is asking me something, I should respond helpfully.'"
        )
    )
    print(
        wrap(
            "\nAn INSTRUCT (or 'chat') model is the same kind of network, "
            "further fine-tuned (typically via supervised fine-tuning + RLHF "
            "or similar) on examples of instructions paired with helpful "
            "responses. That training is what makes it behave like an "
            "assistant instead of an autocomplete engine."
        )
    )
    print(
        wrap(
            "\nGroq hosts specific published checkpoints -- e.g. Meta's "
            "'Llama-3.3-70B-Instruct' (what ChatGroq calls "
            "'openai/gpt-oss-20b') -- note the '-Instruct' suffix Meta "
            "itself uses to mark the chat-tuned variant. Groq generally does "
            "NOT host the corresponding raw base checkpoints for chat use; "
            "the base/instruct split happens upstream, at Meta (or whichever "
            "lab trained the model), before Groq ever serves it."
        )
    )
    print(
        wrap(
            "\n  Prompt:            'The capital of France is'\n"
            "  Base-style completion:   'Paris. It is located on the Seine "
            "river and has a population of...' (just continues the sentence)\n"
            "  Instruct-style completion: 'The capital of France is Paris.' "
            "(answers directly, stops, may add helpful context)"
        )
    )
    print(
        wrap(
            "\nTakeaway: base models are raw continuation engines; instruct "
            "models are tuned to recognize 'this looks like a request' and "
            "respond to it as one. Every model name you can pass to ChatGroq "
            "is an instruct/chat-tuned checkpoint."
        )
    )


# ---------------------------------------------------------------------
# 5. PROMPT ENGINEERING
# ---------------------------------------------------------------------
def demo_prompt_engineering():
    banner("5. PROMPT ENGINEERING")
    bare = "Write a product description."
    engineered = (
        "You are a copywriter for an outdoor gear brand. Write a 2-sentence "
        "product description for a 32oz insulated water bottle. Audience: "
        "day hikers. Tone: confident, no exclamation points. Mention "
        "insulation and durability specifically."
    )
    for label, prompt in (("Bare prompt", bare), ("Engineered prompt", engineered)):
        print(f"\n--- {label} ---")
        print(wrap(prompt))
        resp = call(prompt, max_tokens=120, temperature=0.7)
        if resp:
            print("->")
            print_response(resp)
        else:
            canned = {
                "Bare prompt": "[canned] Sure! Here's a product "
                "description... (generic, needs follow-up questions "
                "about the actual product)",
                "Engineered prompt": "[canned] Built for the trail, this "
                "32oz bottle keeps drinks cold for 24 hours and hot for "
                "12, wrapped in a dent-resistant stainless shell that "
                "shrugs off drops.",
            }[label]
            print("-> " + canned)
    print(
        wrap(
            "\nThe difference is entirely in the input: specifying role, "
            "audience, format, length, and constraints turns a vague request "
            "into a predictable, on-brief output -- no model change required."
        )
    )


# ---------------------------------------------------------------------
# 6. SYSTEM INSTRUCTIONS
# ---------------------------------------------------------------------
def demo_system_instructions():
    banner("6. SYSTEM INSTRUCTIONS")
    system = (
        "You are a terse technical reviewer. Always answer in at "
        "most 2 bullet points. Never use adjectives."
    )
    user_msg = "What do you think of using global variables in Python?"
    print("System prompt:")
    print(wrap(system))
    print("\nUser message:")
    print(wrap(user_msg))
    resp = call(user_msg, system=system, max_tokens=100)
    print("\n->")
    if resp:
        print_response(resp)
    else:
        print(
            "[canned] - Global state makes bugs harder to trace.\n"
            "- Prefer passing values explicitly or using a class."
        )
    print(
        wrap(
            "\nIn LangChain this is a SystemMessage placed first in the "
            "messages list passed to .invoke(). It sets standing behavior for "
            "the whole conversation (persona, tone, constraints, output "
            "rules) separately from the user's actual message -- the right "
            "place for rules that shouldn't be repeated on every turn."
        )
    )


# ---------------------------------------------------------------------
# 7. FORCING JSON OUTPUT
# ---------------------------------------------------------------------
def demo_json_output():
    banner("7. FORCING JSON OUTPUT")
    system = (
        "You extract structured data. Respond with ONLY a JSON object -- "
        "no prose, no markdown fences, no explanation. Schema:\n"
        '{"name": string, "age": number, "city": string}'
    )
    user_msg = "Maria is 29 and just moved to Lisbon for a new job."
    print("System prompt enforces schema + 'JSON only, no prose'.")
    print("User message:", user_msg)

    raw = None
    if HAVE_KEY:
        try:
            # Groq's OpenAI-compatible API supports JSON mode on many
            # models -- LangChain exposes it via model_kwargs on ChatGroq.
            # This makes the API itself reject/repair non-JSON output,
            # which is stronger than prompt-only instructions alone.
            chat = get_chat(
                temperature=0,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            resp = chat.invoke(build_messages(user_msg, system=system))
            raw = resp.content
        except Exception as e:
            print(
                f"[JSON mode call failed: {e} -- falling back to plain "
                "prompt-only call]"
            )
            resp = call(user_msg, system=system, temperature=0)
            raw = resp.content if resp else None

    if raw is None:
        raw = '{"name": "Maria", "age": 29, "city": "Lisbon"}'
        print("[canned example]")

    print("\nRaw model output:", raw)
    try:
        parsed = json.loads(raw.strip().strip("`"))
        print("Parsed successfully:", parsed)
    except json.JSONDecodeError as e:
        print(
            f"Failed to parse JSON ({e}) -- this is why real pipelines "
            "add retry/repair logic or use tool calling instead (see "
            "section 8), which guarantees valid JSON by construction."
        )

    print(
        wrap(
            "\nThree ways to force JSON with ChatGroq / LangChain, in "
            "increasing order of reliability:\n"
            "  a) Prompt-level: instruct the model to emit JSON only, then "
            "parse defensively. Works most of the time, not guaranteed.\n"
            "  b) JSON mode: pass model_kwargs={'response_format': "
            "{'type': 'json_object'}} (shown above) -- the API itself "
            "constrains output to valid JSON, on models that support it.\n"
            "  c) Tool/function calling: define a schema as a 'tool' (or use "
            "LangChain's .with_structured_output(YourPydanticModel)) and let "
            "the model fill it in as a structured, schema-validated call. "
            "See the next section -- this is the most reliable approach."
        )
    )


# ---------------------------------------------------------------------
# 8 & 9. FUNCTION CALLING CORE + PULMOCK INVENTORY CAPSTONE
# ---------------------------------------------------------------------
# "Pulmock" -- a trivial in-memory inventory database standing in for a
# real API. Every "function" below is what a real backend endpoint would
# do; the LLM's job is just to figure out WHICH one to call and with
# WHAT arguments, based on a natural-language request.

PULMOCK_DB = {
    "widgets": {"qty": 120, "location": "Warehouse A"},
    "gadgets": {"qty": 45, "location": "Warehouse B"},
    "gizmos": {"qty": 0, "location": "Warehouse A"},
}


def _pulmock_tools():
    """
    Defined inside a function (rather than at import time) so this file
    still imports cleanly even when langchain_core isn't installed --
    the @tool decorator requires it.
    """

    @tool
    def check_stock(item: str) -> dict:
        """Look up quantity and location for one inventory item."""
        item = item.lower()
        if item not in PULMOCK_DB:
            return {"error": f"'{item}' not found in inventory"}
        return {"item": item, **PULMOCK_DB[item]}

    @tool
    def add_stock(item: str, quantity: int) -> dict:
        """Add units of an item to inventory (receiving new stock)."""
        item = item.lower()
        PULMOCK_DB.setdefault(item, {"qty": 0, "location": "Unassigned"})
        PULMOCK_DB[item]["qty"] += quantity
        return {"item": item, "new_qty": PULMOCK_DB[item]["qty"]}

    @tool
    def remove_stock(item: str, quantity: int) -> dict:
        """Remove units of an item from inventory (shipping/selling stock)."""
        item = item.lower()
        if item not in PULMOCK_DB:
            return {"error": f"'{item}' not found in inventory"}
        if PULMOCK_DB[item]["qty"] < quantity:
            return {
                "error": f"cannot remove {quantity}, only "
                f"{PULMOCK_DB[item]['qty']} in stock"
            }
        PULMOCK_DB[item]["qty"] -= quantity
        return {"item": item, "new_qty": PULMOCK_DB[item]["qty"]}

    @tool
    def list_inventory() -> dict:
        """List every item currently in inventory with quantity and location."""
        return PULMOCK_DB

    tools = [check_stock, add_stock, remove_stock, list_inventory]
    dispatch = {t.name: t for t in tools}
    return tools, dispatch


PULMOCK_SYSTEM = (
    "You are Pulmock's inventory assistant. Use the provided tools to "
    "answer inventory questions and perform stock changes. Always call a "
    "tool rather than guessing numbers. After you get a tool result, give "
    "the user a short, plain-English confirmation."
)


def run_pulmock_agent(user_message: str):
    """
    Full function-calling loop with LangChain + ChatGroq:
      1. Bind the tool schemas to the chat model (model.bind_tools(...)).
      2. Send NL request; model responds with .tool_calls (structured args).
      3. We dispatch each call to the real Python function ('dummy API').
      4. We append a ToolMessage with the result for each call.
      5. Model produces the final natural-language answer.
    """
    print(f"\nUser: {user_message}")

    if not HAVE_KEY:
        print("[no GROQ_API_KEY -- simulating the loop for 'check stock on widgets']")
        print('[simulated tool_use] check_stock({"item": "widgets"})')
        print(
            f"[Pulmock API result] {{'item': 'widgets', **PULMOCK_DB['widgets']}} "
            f"-> {PULMOCK_DB['widgets']}"
        )
        print("Assistant: You have 120 widgets in stock, stored in " "Warehouse A.")
        return

    tools, dispatch = _pulmock_tools()
    chat = get_chat(temperature=0, max_tokens=300)
    chat_with_tools = chat.bind_tools(tools)

    messages = [
        SystemMessage(content=PULMOCK_SYSTEM),
        HumanMessage(content=user_message),
    ]

    ai_msg = chat_with_tools.invoke(messages)
    messages.append(ai_msg)

    if not ai_msg.tool_calls:
        print_response(ai_msg)
        return

    for tc in ai_msg.tool_calls:
        tool_fn = dispatch.get(tc["name"])
        result = tool_fn.invoke(tc["args"]) if tool_fn else {"error": "unknown tool"}
        print(f"[tool_use] {tc['name']}({json.dumps(tc['args'])})")
        print(f"[Pulmock API result] {result}")
        messages.append(ToolMessage(content=json.dumps(result), tool_call_id=tc["id"]))

    final = chat_with_tools.invoke(messages)
    print("Assistant:", end=" ")
    print_response(final)


def demo_function_calling():
    banner("8. FUNCTION CALLING CORE")
    print(
        wrap(
            "Function calling ('tool use') lets the model choose, from a set "
            "of described functions, which one to invoke and with what "
            "arguments -- as structured, schema-validated output instead of "
            "free text. You then run the real function yourself; the model "
            "never executes anything directly."
        )
    )
    print(
        wrap(
            "\nWith LangChain, you define tools with the @tool decorator "
            "(the docstring becomes the description, type hints become the "
            "JSON schema), bind them to the chat model with "
            "model.bind_tools([...]), and call .invoke(). The returned "
            "AIMessage carries a .tool_calls list of {name, args, id} dicts "
            "instead of, or alongside, plain text."
        )
    )
    print(
        wrap(
            "\nThe loop is always: bind tools -> model picks one + fills "
            "arguments -> you run the real function -> append a ToolMessage "
            "with the result -> model writes the final answer. See section 9 "
            "for this running end-to-end against the Pulmock inventory API."
        )
    )


def demo_pulmock_capstone():
    banner("9. CAPSTONE: NATURAL LANGUAGE -> PULMOCK INVENTORY API")
    print(
        wrap(
            "Pulmock is a toy in-memory inventory 'database' "
            f"(currently: {PULMOCK_DB}). The tools below are what a real "
            "warehouse API would expose; the model's only job is picking the "
            "right one and extracting arguments from plain English."
        )
    )
    examples = [
        "How many widgets do we have?",
        "We just received 30 more gadgets.",
        "Ship out 10 gizmos.",
        "Give me the full inventory list.",
    ]
    for ex in examples:
        run_pulmock_agent(ex)

    print(
        wrap(
            "\nTry it interactively: import this module and call "
            "run_pulmock_agent('your request here'), or run this script with "
            "the 'pulmock' argument for a REPL."
        )
    )


def interactive_pulmock():
    banner("PULMOCK -- INTERACTIVE MODE (type 'quit' to exit)")
    while True:
        try:
            msg = input("\nYou: ").strip()
        except EOFError:
            break
        if msg.lower() in ("quit", "exit"):
            break
        if msg:
            run_pulmock_agent(msg)


# ---------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------
DEMOS = {
    "tokenization": demo_tokenization,
    "contextwindow": demo_context_window,
    "temperature": demo_temperature,
    "baseinstruct": demo_base_vs_instruct,
    "promptengineering": demo_prompt_engineering,
    "systeminstructions": demo_system_instructions,
    "jsonoutput": demo_json_output,
    "functioncalling": demo_function_calling,
    "capstone": demo_pulmock_capstone,
}


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    # if ChatGroq is None:
    #     print(
    #         "langchain_groq isn't installed. Run:\n"
    #         "  pip install langchain langchain-groq langchain-core "
    #         "--break-system-packages"
    # )
    if not HAVE_KEY:
        print(
            "NOTE: GROQ_API_KEY not set. Demos will run with canned "
            "example output so you can still see the mechanics. Set "
            "the env var and re-run for live model calls."
        )

    if arg == "pulmock":
        interactive_pulmock()
        return

    if arg == "all":
        for fn in DEMOS.values():
            fn()
        return

    fn = DEMOS.get(arg)
    if not fn:
        print(f"Unknown demo '{arg}'. Options: all, pulmock, " f"{', '.join(DEMOS)}")
        sys.exit(1)
    fn()


if __name__ == "__main__":
    main()
