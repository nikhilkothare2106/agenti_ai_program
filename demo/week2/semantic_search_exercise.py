from collections import Counter
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from model_config import embedding_model

SENTENCES_BY_CLUSTER = {
    "leave policy": [
        "Employees request annual leave through the HR portal.",
        "The manager approves annual leave after checking team coverage.",
        "Policy HR-114-B says unused leave does not carry forward.",
        "The manager rejects annual leave after checking team coverage.",
        "Unused leave carries forward to the next calendar year.",
        "Sick leave can be used when an employee is unwell.",
        "Tell your manager when you need an unexpected personal day.",
        "Parental leave begins after the required HR paperwork is filed.",
        "A public holiday does not count against annual leave balance.",
        "Employees should give two weeks notice for planned leave.",
        "Check your leave balance in the employee self-service portal.",
        "Medical appointments may qualify for sick leave.",
        "Unpaid leave requires approval from the department director.",
        "The team calendar shows approved employee absences.",
        "Bereavement leave is available after a close family loss.",
        "New employees accrue vacation time each month.",
        "Cancel a leave request in the HR portal before it is approved.",
        "The leave policy explains eligibility for a career break.",
        "Half-day leave requests are entered as four hours.",
        "Submit a doctor's note when HR requests documentation for sick leave.",
    ],
    "payroll": [
        "Paychecks are deposited into the bank account on file.",
        "Update your tax withholding in the payroll portal.",
        "Monthly pay statements are available online after payday.",
        "Contact payroll if your salary deposit is missing.",
        "Overtime hours must be approved before payroll closes.",
        "A bonus appears as a separate line on the pay statement.",
        "Claims above 50,000 dollars require finance director approval.",
        "Payroll deductions include health insurance premiums.",
        "Employees can change direct deposit details in self-service.",
        "The payroll team corrects errors in the next pay cycle.",
        "Contractors submit invoices instead of receiving employee paychecks.",
        "A year-end tax form is posted in the payroll portal.",
        "Hourly staff record their shifts on a weekly timesheet.",
        "A promotion may change your salary starting next month.",
        "Payroll needs your updated mailing address for tax documents.",
        "Commission payments are calculated after each sales quarter.",
        "The pay calendar lists every deposit date for the year.",
        "Claims below 50,000 dollars may be approved by a manager.",
        "Final wages are processed after an employee leaves the company.",
        "A late timesheet can delay payment for recorded hours.",
    ],
    "IT support": [
        "Open a help desk ticket when your laptop will not start.",
        "Reset your network password from the account recovery page.",
        "The IT team installs approved software on company computers.",
        "Connect to the secure VPN before accessing internal files.",
        "Report a phishing email using the mail security button.",
        "Request a replacement keyboard through the service portal.",
        "Restart the printer if it stops responding to print jobs.",
        "Two-factor authentication protects access to your work account.",
        "IT can restore files from the company backup system.",
        "A software update may require restarting your computer.",
        "Contact the help desk if your monitor shows no signal.",
        "Employees should lock their screens before leaving a desk.",
        "The service portal shows the status of open support tickets.",
        "Request access to a shared folder from its team owner.",
        "A damaged company phone should be reported to IT support.",
        "Use the approved video application for online meetings.",
        "IT removes access when an employee changes departments.",
        "Clear your browser cache if the internal site will not load.",
        "The help desk can troubleshoot a slow wireless connection.",
        "Never share your login code with someone claiming to be IT.",
    ],
    "food": [
        "The cafeteria serves lunch from eleven thirty until two.",
        "Vegetarian meals are marked on the weekly menu.",
        "Use the kitchen refrigerator for labeled personal lunches.",
        "The office coffee machine is beside the main kitchen.",
        "Employees can order sandwiches for the team meeting.",
        "Please list common allergens when booking catered food.",
        "Fresh fruit is delivered to the break room each Tuesday.",
        "The cafe accepts meal vouchers at the checkout counter.",
        "Store leftovers in a sealed container in the refrigerator.",
        "The caterer provides vegan options for company events.",
        "Bring a reusable mug for coffee or tea in the kitchen.",
        "The lunch menu includes soup, salad, and hot entrees.",
        "Clean the microwave after heating food in the break room.",
        "Order breakfast pastries before the morning staff meeting.",
        "The office kitchen has oat milk next to the regular milk.",
        "Ask the event organizer about gluten-free meal choices.",
        "The cafeteria closes early on Friday afternoons.",
        "Label your lunch with your name and the date.",
        "Employees pay for snacks at the self-service pantry.",
        "A water dispenser is available near the dining area.",
    ],
    "travel": [
        "Book business flights through the company's travel website.",
        "Get manager approval before reserving an overseas hotel.",
        "Save taxi receipts for your work trip expense report.",
        "The travel desk can help change a flight reservation.",
        "Check passport expiration dates before international travel.",
        "Use the corporate rate when booking a rental car.",
        "Submit travel expenses within ten days of returning.",
        "The company reimburses reasonable baggage fees for work trips.",
        "Download your boarding pass before leaving for the airport.",
        "Travel insurance covers eligible emergencies during business trips.",
        "Choose a hotel near the conference venue when possible.",
        "A delayed flight should be reported to your meeting host.",
        "Keep meal receipts when traveling for a client visit.",
        "The travel policy explains which train fares are reimbursable.",
        "Check visa requirements before booking an international trip.",
        "Employees should use public transit when practical on work trips.",
        "Contact the travel desk for help with a canceled reservation.",
        "Enter the project code on each business travel expense.",
        "A checked bag may be reimbursed for a week-long work trip.",
        "SkyTrail is the company portal for managing business flight reservations.",
    ],
}

SENTENCES = [
    (cluster, sentence)
    for cluster, sentences in SENTENCES_BY_CLUSTER.items()
    for sentence in sentences
]
TEXTS = [sentence for _, sentence in SENTENCES]
CLUSTERS = [cluster for cluster, _ in SENTENCES]
bm25 = BM25Okapi([re.findall(r"[a-z0-9-]+", text.lower()) for text in TEXTS])


def search(query: str, k: int = 5) -> list[tuple[str, float, str]]:
    """Return the k most similar sentences as (text, cosine score, cluster)."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")

    query_vector = np.asarray(embedding_model.embed_query(query), dtype=float)
    scores = cosine_similarity(query_vector.reshape(1, -1), embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:k]
    return [
        (TEXTS[index], float(scores[index]), CLUSTERS[index]) for index in top_indices
    ]


def keyword_search(query: str, k: int = 5) -> list[tuple[str, float, str]]:
    """Return BM25-ranked sentences to compare exact-term retrieval with vectors."""
    tokens = re.findall(r"[a-z0-9-]+", query.lower())
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    return [
        (TEXTS[index], float(scores[index]), CLUSTERS[index])
        for index in top_indices
        if scores[index] > 0
    ]


def main() -> None:
    cluster_counts = Counter(CLUSTERS)
    if len(TEXTS) != 100 or set(cluster_counts.values()) != {20}:
        raise ValueError(
            f"Expected 100 sentences across five equal clusters: {cluster_counts}"
        )

    global embeddings
    embeddings = np.asarray(embedding_model.embed_documents(TEXTS), dtype=float)
    print(f"Embedding array shape: {embeddings.shape}")
    print(
        "Shape dimensions: 100 rows, one row per sentence; "
        f"{embeddings.shape[1]} columns, one value per embedding feature."
    )
    print("Cluster counts:", dict(cluster_counts))

    similarity_matrix = cosine_similarity(embeddings)
    print(f"Cosine similarity matrix shape: {similarity_matrix.shape}")
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        similarity_matrix,
        cmap="mako",
        vmin=-1,
        vmax=1,
        xticklabels=False,
        yticklabels=False,
    )
    plt.title("Cosine similarity across 100 workplace sentences")
    plt.xlabel("Sentence index, grouped by topic")
    plt.ylabel("Sentence index, grouped by topic")
    plt.tight_layout()
    # heatmap_path = Path(__file__).with_name("cosine_similarity_heatmap.png")
    # plt.savefig(heatmap_path, dpi=160)
    plt.close()
    # print(f"Saved cosine similarity heatmap to {heatmap_path}")

    queries = [
        "How do I ask for time off next month?",
        "Why has my paycheck not arrived?",
        "I cannot connect to the company network.",
        "What vegetarian options are available for lunch?",
        "How can I change a flight for a work trip?",
        "Where can I update my bank details?",
        "Can I take a day off for a medical appointment?",
        "Who can help me install approved software?",
        "Can the office provide gluten-free catering?",
        "Which receipts do I need after traveling for work?",
    ]
    print("\nTen search queries (top 5 results each):")
    for query in queries:
        print(f"\nQuery: {query}")
        for text, score, cluster in search(query, k=5):
            print(f"  [{score:.3f}] ({cluster}) {text}")

    # These deliberately mix topic cues; compare the top hit with the intended task.
    failure_probes = [
        (
            "What is policy HR-114-B?",
            "leave policy",
            "Policy HR-114-B says unused leave does not carry forward.",
            "The identifier itself has little semantic meaning. BM25 can match HR-114-B exactly while the vector may rank a generic leave-policy sentence higher.",
        ),
        (
            "leave that is not carried forward",
            "leave policy",
            "Policy HR-114-B says unused leave does not carry forward.",
            "Negation can have a small effect on embedding similarity, bringing the opposite carryover statement close to the intended one.",
        ),
        (
            "claims above 50,000",
            "payroll",
            "Claims above 50,000 dollars require finance director approval.",
            "Embeddings represent the threshold as text; they do not reliably reason about the numeric comparison or preserve an exact cutoff.",
        ),
        (
            "Who manages reservations in SkyTrail?",
            "travel",
            "SkyTrail is the company portal for managing business flight reservations.",
            "A rare product name has little learned meaning. Exact-term search can still locate the one sentence containing SkyTrail.",
        ),
        (
            "Does the manager reject annual leave after checking team coverage?",
            "leave policy",
            "The manager rejects annual leave after checking team coverage.",
            "The approve and reject sentences share nearly all their words, so their vectors can be very close even though the decisions are opposites.",
        ),
        (
            "I need a day off after my flight and also need reimbursement for the trip.",
            "leave policy",
            "Tell your manager when you need an unexpected personal day.",
            "The query combines leave and travel. Embeddings favored the travel reimbursement wording, while BM25 matched the exact day-off terms.",
        ),
        (
            "My payroll login is broken; who can reset it?",
            "IT support",
            "Reset your network password from the account recovery page.",
            "Payroll context outweighed the login-reset intent, so the vector result described a salary deposit instead of account recovery.",
        ),
        (
            "I am going on leave; where do I submit reimbursement for flights?",
            "leave policy",
            "Policy HR-114-B says unused leave does not carry forward.",
            "The travel reimbursement phrase dominated the leave-policy intent and pulled the result into travel.",
        ),
    ]
    print(
        "\nDeliberate failure probes (five failure types plus three observed misses):"
    )
    for query, intended_cluster, expected_text, diagnosis in failure_probes:
        top_text, score, returned_cluster = search(query, k=1)[0]
        keyword_results = keyword_search(query, k=1)
        keyword_text = keyword_results[0][0] if keyword_results else "No keyword match"
        failed = top_text != expected_text
        outcome = "EMBEDDING MISS" if failed else "expected sentence ranked first"
        print(f"\n[{outcome}] Query: {query}")
        print(f"  Expected ({intended_cluster}): {expected_text}")
        print(
            f"  Embedding returned ({returned_cluster}, cosine={score:.3f}): {top_text}"
        )
        print(f"  BM25 keyword result: {keyword_text}")
        print(f"  What broke: {diagnosis}")
        if failed and keyword_text == expected_text:
            print("  Keyword search would have got this; embeddings did not.")

    coordinates = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=42,
    ).fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    palette = dict(zip(SENTENCES_BY_CLUSTER, sns.color_palette("colorblind", 5)))
    for cluster, color in palette.items():
        mask = np.asarray(CLUSTERS) == cluster
        plt.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            label=cluster,
            color=color,
            alpha=0.8,
            s=55,
        )
    plt.title("t-SNE view of workplace sentence embeddings")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend(title="Topic")
    plt.tight_layout()
    # tsne_path = Path(__file__).with_name("tsne_clusters.png")
    # plt.savefig(tsne_path, dpi=160)
    plt.close()
    # print(f"Saved t-SNE plot to {tsne_path}")


if __name__ == "__main__":
    main()
