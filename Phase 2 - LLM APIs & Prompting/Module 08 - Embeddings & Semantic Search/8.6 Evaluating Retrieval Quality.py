import numpy as np
from dataclasses import dataclass


@dataclass
class RetrievalEvalCase:
    query: str
    relevant_doc_ids: list[str]  # ground-truth relevant documents


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:
    """Fraction of top-k results that are relevant."""

    top_k = retrieved_ids[:k]

    hits = sum(
        1
        for doc_id in top_k
        if doc_id in relevant_ids
    )

    return hits / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:
    """Fraction of all relevant docs found in top-k."""

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    hits = sum(
        1
        for doc_id in top_k
        if doc_id in relevant_ids
    )

    return hits / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str]
) -> float:
    """MRR: reciprocal of the rank of the first relevant result."""

    for rank, doc_id in enumerate(
        retrieved_ids,
        start=1
    ):
        if doc_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(
    store,  # VectorStore from section 8.4
    eval_cases: list[RetrievalEvalCase],
    k: int = 5,
) -> dict:
    """Run all eval cases and return aggregate metrics."""

    p_scores = []
    r_scores = []
    mrr_scores = []

    for case in eval_cases:

        results = store.search(
            case.query,
            k=k
        )

        retrieved_ids = [
            r.document.id
            for r in results
        ]

        p_scores.append(
            precision_at_k(
                retrieved_ids,
                case.relevant_doc_ids,
                k
            )
        )

        r_scores.append(
            recall_at_k(
                retrieved_ids,
                case.relevant_doc_ids,
                k
            )
        )

        mrr_scores.append(
            mean_reciprocal_rank(
                retrieved_ids,
                case.relevant_doc_ids
            )
        )

    return {
        f"precision@{k}": round(
            float(np.mean(p_scores)),
            4
        ),
        f"recall@{k}": round(
            float(np.mean(r_scores)),
            4
        ),
        "MRR": round(
            float(np.mean(mrr_scores)),
            4
        ),
    }


# Using the VectorStore and CORPUS from section 8.4
eval_cases = [
    RetrievalEvalCase(
        "How does RAG work?",
        ["d01", "d08"]
    ),

    RetrievalEvalCase(
        "What are vector databases?",
        ["d02", "d06"]
    ),

    RetrievalEvalCase(
        "How do agents use language models?",
        ["d10"]
    ),

    RetrievalEvalCase(
        "What is fine-tuning?",
        ["d03"]
    ),

    RetrievalEvalCase(
        "How do transformers model token relationships?",
        ["d09"]
    ),
]


import importlib.util
import sys
import os

print("Loading VectorStore from 8.4 (this may take a few seconds)...")
# Dynamically load the module because the filename has spaces
# Use the same directory as this script so it works regardless of CWD
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "8.4 Semantic Search - Full Pipeline.py")
spec = importlib.util.spec_from_file_location("module_8_4", file_path)
module_8_4 = importlib.util.module_from_spec(spec)
sys.modules["module_8_4"] = module_8_4
# This will execute 8.4 and populate the 'store'
try:
    spec.loader.exec_module(module_8_4)
    store = module_8_4.store

    print("\n--- Running Evaluation ---")
    metrics = evaluate_retrieval(store, eval_cases, k=3)
    print("\nEvaluation Results:")
    print(metrics)

except Exception as e:
    print(f"\n[ERROR] Failed to load store from 8.4: {str(e)}")
    print("Make sure 8.4 Semantic Search - Full Pipeline.py can run successfully first.")
