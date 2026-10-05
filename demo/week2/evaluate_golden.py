import csv
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEMORY_DIR = ROOT / "rag_concepts" / "memory"
sys.path.insert(0, str(MEMORY_DIR))

# Handle missing Vertex AI dependency
try:
    from langchain_community.chat_models.vertexai import ChatVertexAI
except ModuleNotFoundError:
    module = types.ModuleType("langchain_community.chat_models.vertexai")
    module.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules["langchain_community.chat_models.vertexai"] = module

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from langchain_core.messages import HumanMessage, SystemMessage

from main import NO_INFO, SYSTEM_PROMPT, model, similarity_search
from model_config import embedding_model

DATA_DIR = Path(__file__).resolve().parent / "data"
GOLDEN_PATH = DATA_DIR / "golden.csv"
RESULTS_PATH = DATA_DIR / "evaluation_results.md"

METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

RUNS = [
    ("Before", 4),
    ("After", 6),
]


def load_cases():
    with open(GOLDEN_PATH, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def answer_case(question, k):
    documents = similarity_search(question, k=k)

    contexts = [doc.page_content for doc in documents if doc.page_content.strip()]

    if not contexts:
        return NO_INFO, contexts

    context = "\n\n".join(contexts)

    response = model.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"),
        ]
    )

    answer = (
        response.content if isinstance(response.content, str) else str(response.content)
    )

    return answer, contexts


def evaluate_run(cases, k):
    answers = []

    for case in cases:
        answer, contexts = answer_case(case["question"], k)

        answers.append(
            {
                **case,
                "answer": answer,
                "contexts": contexts,
            }
        )

    # Check how many unanswerable questions were correctly refused
    unanswerable = [case for case in cases if case["answerable"] == "false"]

    refusal_correct = sum(
        answer["answer"].strip() == NO_INFO
        for answer in answers
        if answer["answerable"] == "false"
    )

    # Only answerable questions are evaluated by RAGAS
    ragas_data = [
        {
            "question": item["question"],
            "answer": item["answer"],
            "contexts": item["contexts"],
            "ground_truth": item["ground_truth"],
        }
        for item in answers
        if item["answerable"] == "true"
    ]

    scores = evaluate(
        Dataset.from_list(ragas_data),
        metrics=METRICS,
        llm=model,
        embeddings=embedding_model,
        raise_exceptions=True,
    ).to_pandas()

    metric_scores = {
        metric.name: float(scores[metric.name].mean()) for metric in METRICS
    }

    return metric_scores, refusal_correct, len(unanswerable)


def main():
    cases = load_cases()

    # Validate golden dataset
    answerable_count = sum(case["answerable"] == "true" for case in cases)

    if len(cases) != 20 or answerable_count != 15:
        raise ValueError(
            "Expected exactly 15 answerable and 5 unanswerable golden cases"
        )

    results = []

    # Evaluate both configurations
    for label, k in RUNS:
        scores, correct, total = evaluate_run(cases, k)
        results.append((label, k, scores, correct, total))

    # Create markdown table
    metric_names = [metric.name for metric in METRICS]

    headers = [
        "Run",
        "k",
        *metric_names,
        "Refusals correct",
    ]

    rows = []

    for label, k, scores, correct, total in results:
        rows.append(
            [
                label,
                str(k),
                *(f"{scores[name]:.3f}" for name in metric_names),
                f"{correct}/{total}",
            ]
        )

    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

    for row in rows:
        markdown += "| " + " | ".join(row) + " |\n"

    RESULTS_PATH.write_text(markdown, encoding="utf-8")

    print(markdown)
    print("Changed only k: 4 -> 6; chunk size and reranking were unchanged.")


if __name__ == "__main__":
    main()
