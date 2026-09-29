import json
import os
import sys

from langchain_core.messages import HumanMessage, SystemMessage

from model_config import model

TRANSCRIPT = (
    "Okay so let's recap. Sarah, can you get the API rate-limiting fix out by "
    "Thursday? Also I think we still need someone to update the pricing page — "
    "Jon, that's you, no rush but let's say end of next week. Oh and I almost "
    "forgot, we need to send the security audit doc to the client, that one's "
    "actually urgent, can someone do that today. I'll handle the follow-up email "
    "to the vendor myself, probably tomorrow."
)

# ---------------------------------------------------------------- Version 1
# One user message, no system prompt.
V1_USER = (
    "Extract the action items from this meeting transcript. "
    "Return them as JSON: an array of objects with the fields {assignee, task, deadline}.\n\n"
    'Transcript:\n"' + TRANSCRIPT + '"'
)

# ---------------------------------------------------------------- Version 2
# Role + exact schema + explicit rules for missing/ambiguous fields;
# the user message contains only the transcript.
V2_SYSTEM = """You are an action-item extractor for meeting transcripts.

OUTPUT FORMAT
Return a JSON array and nothing else: no prose, no markdown fences, no comments.
Each element is an object with exactly these three keys, in this order, and every value is a string:
  "assignee": who is responsible
  "task":     what must be done
  "deadline": when it is due

RULES
1. Include every action item, in the order it appears in the transcript. Never merge, split, or drop items, and never invent items that are not in the transcript.
2. assignee:
   - If a person is addressed or named as the owner, use the name exactly as spoken.
   - If the speaker takes the task themselves in the first person ("I'll handle"), use "SPEAKER". Do not guess the speaker's name.
   - If nobody is named and nobody claims it ("can someone do that"), use "UNASSIGNED". Never guess or invent a name.
3. task: an imperative phrase of at most 12 words, starting with a verb. Do not put names or deadlines inside it.
4. deadline: copy the timing as it was expressed ("Thursday", "end of next week", "today", "tomorrow"). Do not convert it to a calendar date, because the meeting date is not given. If no timing is stated, use "UNSPECIFIED".
5. Remarks about priority or urgency ("urgent", "no rush") are not deadlines and must not appear in any field.
6. If the transcript contains no action items, return [].
"""
V2_USER = TRANSCRIPT  # minimal user message: only the transcript


# ------------------------------------------------------------------ gold answer
GOLD = [
    {
        "assignee": "Sarah",
        "task": "Fix API rate limiting",
        "deadline": "Thursday",
    },
    {
        "assignee": "Jon",
        "task": "Update pricing page",
        "deadline": "end of next week",
    },
    {
        "assignee": "UNASSIGNED",
        "task": "Send security audit doc to client",
        "deadline": "today",
    },
    {
        "assignee": "SPEAKER",
        "task": "Send follow-up email to vendor",
        "deadline": "tomorrow",
    },
]


def judge_response(response_text, gold_answer):
    """Ask the configured model to judge a response against the gold answer."""
    system_prompt = """
You are a strict but fair evaluator.
Compare the model response against the expected gold answer.

Rules:
- Score from 0 to 10.
- 10 means every item matches the gold answer in meaning.
- Wording differences are fine if the meaning is the same.
- Do NOT give a low score just because the task text is rephrased.
- Give a low score or 0 only when assignee, task, or deadline is materially wrong.
- Wrong values, missing items, wrong order, or bad structure should reduce the score.
- If the response is valid JSON, judge it by meaning, not exact wording.
- Return only valid JSON with exactly this shape:
{
  "score": 0,
  "passed": false,
  "reason": "short explanation"
}

Scoring guide:
- 10: all items match gold in meaning and order; wording can differ.
- 8-9: same items and meaning; small wording differences only.
- 6-7: mostly correct but a few task phrases are less precise.
- 3-5: one item has a wrong assignee, wrong task meaning, or wrong deadline.
- 0-2: multiple wrong values, missing items, invalid JSON, or clearly wrong output.
- passed = true only when score >= 8.
"""

    user_prompt = f"""
Gold answer:
{json.dumps(gold_answer, ensure_ascii=False, indent=2)}

Model response:
{response_text}
"""

    result = model.invoke(
        [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)],
        config={"temperature": 0},
    )
    content = result.content
    if isinstance(content, list):
        cleaned = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    else:
        cleaned = str(content)

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except (TypeError, ValueError):
        pass

    return {"score": 0, "passed": False, "reason": "Judge output was not valid JSON."}


# --------------------------------------------------------------- model calls
def call_model(model_client, system, user, temperature):
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=user))

    response = model_client.invoke(messages)
    content = response.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content if isinstance(content, str) else str(content)


# ------------------------------------------------------------------ simple validation
def selftest():
    """Quick sanity check with canned responses."""
    good = (
        '[{"assignee":"Sarah","task":"Fix API rate limiting","deadline":"Thursday"},'
        '{"assignee":"Jon","task":"Update pricing page","deadline":"end of next week"},'
        '{"assignee":"UNASSIGNED","task":"Send security audit doc to client","deadline":"today"},'
        '{"assignee":"SPEAKER","task":"Send follow-up email to vendor","deadline":"tomorrow"}]'
    )
    bad = good.replace('"UNASSIGNED"', '"Sarah"')

    for label, text in [("good answer", good), ("bad answer", bad)]:
        result = judge_response(text, GOLD)
        print(
            f"{label}: score={result.get('score', 0)}, passed={result.get('passed', False)}"
        )
        print(result.get("reason", ""))


# ---------------------------------------------------------------------- main
def main():
    variants = [
        ("V1 (no system prompt)", None, V1_USER),
        ("V2 (with system prompt)", V2_SYSTEM, V2_USER),
    ]

    for label, system, user in variants:
        scores = []
        all_responses = []
        all_judgements = []
        for _ in range(3):
            response = call_model(model, system, user, temperature=0.0)
            all_responses.append(response)
            result = judge_response(response, GOLD)
            all_judgements.append(result)
            score = result.get("score", 0)
            scores.append(score)

        average_score = sum(scores) / len(scores) if scores else 0

        print(f"\n## {label}")
        for i, (response, judgement) in enumerate(
            zip(all_responses, all_judgements), start=1
        ):
            print(f"\nRun {i}:\n{response}")
            print("Judge result:")
            print(json.dumps(judgement, ensure_ascii=False, indent=2))
        print(f"\nAverage score: {average_score:.2f}/10")
        print(f"Scores: {scores}")


if __name__ == "__main__":
    main()
