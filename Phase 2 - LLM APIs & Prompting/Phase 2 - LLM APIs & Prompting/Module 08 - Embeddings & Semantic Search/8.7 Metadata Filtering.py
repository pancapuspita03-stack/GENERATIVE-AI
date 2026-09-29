from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


@dataclass
class FilteredDocument:
    id: str
    text: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )
    embedding: Optional[np.ndarray] = field(
        default=None,
        repr=False
    )


def embed_texts(
    texts: list[str]
) -> np.ndarray:

    resp = openai_client.embeddings.create(
        input=texts,
        model="openai/text-embedding-3-small"
    )

    vecs = np.array(
        [
            e.embedding
            for e in sorted(
                resp.data,
                key=lambda x: x.index
            )
        ],
        dtype=np.float32
    )

    norms = np.linalg.norm(
        vecs,
        axis=1,
        keepdims=True
    )

    return vecs / np.where(
        norms == 0,
        1,
        norms
    )


class FilteredVectorStore:

    def __init__(self):
        self._docs: list[FilteredDocument] = []

    def add(
        self,
        docs: list[FilteredDocument]
    ) -> None:

        embeddings = embed_texts(
            [d.text for d in docs]
        )

        for doc, emb in zip(
            docs,
            embeddings
        ):
            doc.embedding = emb
            self._docs.append(doc)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_fn: Optional[
            Callable[[FilteredDocument], bool]
        ] = None,
    ) -> list[tuple[FilteredDocument, float]]:
        """Search with optional metadata filter applied before ranking."""

        # Apply pre-filter
        candidates = (
            self._docs
            if filter_fn is None
            else [
                d
                for d in self._docs
                if filter_fn(d)
            ]
        )

        if not candidates:
            return []

        # Embed query
        q_vec = embed_texts(
            [query]
        )[0]

        # Score
        matrix = np.array(
            [
                d.embedding
                for d in candidates
            ],
            dtype=np.float32
        )

        scores = matrix @ q_vec

        # Top-k
        k = min(
            k,
            len(candidates)
        )

        top_idx = np.argsort(
            scores
        )[::-1][:k]

        return [
            (
                candidates[i],
                float(scores[i])
            )
            for i in top_idx
        ]


# Demo - a mixed corpus with category metadata
docs = [
    FilteredDocument(
        "a1",
        "GPT-4o supports vision and function calling.",
        {
            "category": "openai",
            "year": 2024
        }
    ),

    FilteredDocument(
        "a2",
        "Claude 3.5 Sonnet excels at coding tasks.",
        {
            "category": "anthropic",
            "year": 2024
        }
    ),

    FilteredDocument(
        "a3",
        "GPT-4o-mini is a smaller, cheaper model.",
        {
            "category": "openai",
            "year": 2024
        }
    ),

    FilteredDocument(
        "a4",
        "Claude Opus 4 is Anthropic's most capable model.",
        {
            "category": "anthropic",
            "year": 2025
        }
    ),

    FilteredDocument(
        "a5",
        "GPT-4 Turbo has a 128K context window.",
        {
            "category": "openai",
            "year": 2023
        }
    ),
]


try:
    fstore = FilteredVectorStore()

    fstore.add(
        docs
    )

    # Search across all docs
    print("=== All docs ===")

    results = fstore.search(
        "which model is good at coding?",
        k=3
    )

    for doc, score in results:
        print(
            f" [{score:.4f}] {doc.id}: {doc.text}"
        )

    # Search only Anthropic docs
    print("\n=== Anthropic only ===")

    results = fstore.search(
        "which model is good at coding?",
        k=3,
        filter_fn=lambda d: (
            d.metadata["category"] == "anthropic"
        )
    )

    for doc, score in results:
        print(
            f" [{score:.4f}] {doc.id}: {doc.text}"
        )

except Exception as e:
    print("\n[ERROR] Failed to run Metadata Filtering demo.")
    print(f"Original Error: {str(e)}")