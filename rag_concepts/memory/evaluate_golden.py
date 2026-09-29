import csv
import sys
import types
from pathlib import Path

from datasets import Dataset

_vertex_chat_module = "langchain_community.chat_models.vertexai"
try:
    __import__(_vertex_chat_module, fromlist=["ChatVertexAI"])
except ModuleNotFoundError as error:
    if error.name != _vertex_chat_module:
        raise
    _vertex_chat = types.ModuleType(_vertex_chat_module)
    _vertex_chat.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules[_vertex_chat_module] = _vertex_chat

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

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = ROOT / "data" / "golden.csv"
RESULTS_PATH = ROOT / "data" / "evaluation_results.md"
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
RUNS = [("Before", 4), ("After", 6)]


def load_cases():
    with GOLDEN_PATH.open(newline="", encoding="utf-8") as golden_file:
        return list(csv.DictReader(golden_file))


def answer_case(question, k):
    documents = similarity_search(question, k=k)
    contexts = [
        document.page_content for document in documents if document.page_content.strip()
    ]
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
    unanswerable = [case for case in cases if case["answerable"] == "false"]
    answers = []
    for case in cases:
        answer, contexts = answer_case(case["question"], k)
        answers.append({**case, "answer": answer, "contexts": contexts})

    refusal_correct = sum(
        item["answer"].strip() == NO_INFO
        for item in answers
        if item["answerable"] == "false"
    )
    ragas_rows = [
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
        Dataset.from_list(ragas_rows),
        metrics=METRICS,
        llm=model,
        embeddings=embedding_model,
        raise_exceptions=True,
    ).to_pandas()
    return (
        {metric.name: float(scores[metric.name].mean()) for metric in METRICS},
        refusal_correct,
        len(unanswerable),
    )


def main():
    cases = load_cases()
    if len(cases) != 20 or sum(case["answerable"] == "true" for case in cases) != 15:
        raise ValueError(
            "Expected exactly 15 answerable and 5 unanswerable golden cases"
        )

    results = []
    for label, k in RUNS:
        scores, refusal_correct, refusal_total = evaluate_run(cases, k)
        results.append((label, k, scores, refusal_correct, refusal_total))

    metric_names = [metric.name for metric in METRICS]
    headers = ["Run", "k", *metric_names, "Refusals correct"]
    rows = [
        [
            label,
            str(k),
            *(f"{scores[name]:.3f}" for name in metric_names),
            f"{correct}/{total}",
        ]
        for label, k, scores, correct, total in results
    ]
    markdown = "| " + " | ".join(headers) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    markdown += "\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n"
    RESULTS_PATH.write_text(markdown, encoding="utf-8")
    print(markdown)
    print("Changed only k: 4 -> 6; chunk size and reranking were unchanged.")


if __name__ == "__main__":
    main()
