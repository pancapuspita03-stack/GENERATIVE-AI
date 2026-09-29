import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


# ── Data structures ─────────────────────────────────────────

@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(
        default=None,
        repr=False
    )


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int


# ── Embedding helper ─────────────────────────────────────────

def embed_batch(
    texts: list[str],
    model: str = "openai/text-embedding-3-small"
) -> np.ndarray:
    """Embed texts in a single API call. Returns (n, dim) float32 array."""

    response = openai_client.embeddings.create(
        input=texts,
        model=model
    )

    vectors = sorted(
        response.data,
        key=lambda e: e.index
    )

    return np.array(
        [v.embedding for v in vectors],
        dtype=np.float32
    )


# ── Simple in-memory vector store ────────────────────────────

class VectorStore:
    """
    In-memory vector store for semantic search.
    Suitable for corpora up to ~100k documents.
    For larger collections use ChromaDB, Pinecone, or pgvector
    (covered in Phase 3).
    """

    def __init__(
        self,
        embed_model: str = "openai/text-embedding-3-small"
    ):
        self.embed_model = embed_model
        self._documents: list[Document] = []
        self._matrix: Optional[np.ndarray] = None  # (n, dim) normalised

    def add_documents(
        self,
        documents: list[Document]
    ) -> None:
        """Embed and index a list of documents."""

        texts = [d.text for d in documents]

        vectors = embed_batch(
            texts,
            model=self.embed_model
        )

        # Normalise for fast cosine similarity via dot product
        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True
        )

        norms = np.where(
            norms == 0,
            1,
            norms
        )

        normed = (
            vectors / norms
        ).astype(np.float32)

        for doc, vec in zip(
            documents,
            normed
        ):
            doc.embedding = vec
            self._documents.append(doc)

        # Rebuild the full matrix
        self._matrix = np.array(
            [d.embedding for d in self._documents],
            dtype=np.float32
        )

        print(
            f"Index now contains {len(self._documents)} documents."
        )

    def search(
        self,
        query: str,
        k: int = 5
    ) -> list[SearchResult]:
        """Return the k most similar documents for a query string."""

        if (
            self._matrix is None
            or len(self._documents) == 0
        ):
            raise RuntimeError(
                "No documents indexed yet."
            )

        # Embed and normalise query
        q_vec = embed_batch(
            [query],
            model=self.embed_model
        )[0]

        q_norm = np.linalg.norm(q_vec)

        if q_norm == 0:
            return []

        q_vec = (
            q_vec / q_norm
        ).astype(np.float32)

        # Cosine similarities:
        # matrix @ vector → shape (n,)
        scores = self._matrix @ q_vec

        # Top-k indices (descending)
        k = min(
            k,
            len(self._documents)
        )

        top_idx = np.argsort(
            scores
        )[::-1][:k]

        return [
            SearchResult(
                document=self._documents[int(i)],
                score=float(scores[i]),
                rank=rank + 1,
            )
            for rank, i in enumerate(top_idx)
        ]

    @property
    def size(self) -> int:
        return len(self._documents)


# ── Demo ─────────────────────────────────────────────────────

CORPUS = [
    Document(
        "d01",
        "Retrieval-Augmented Generation (RAG) combines information retrieval with language model generation to answer questions using external knowledge."
    ),
    Document(
        "d02",
        "Vector databases store high-dimensional embeddings and enable fast approximate nearest-neighbour search using algorithms like HNSW and IVF."
    ),
    Document(
        "d03",
        "Fine-tuning adapts a pre-trained language model to a specific task by continuing training on a curated dataset with task-specific examples."
    ),
    Document(
        "d04",
        "Prompt engineering involves designing and optimising input prompts to guide language models toward producing the desired output."
    ),
    Document(
        "d05",
        "LangChain is a Python framework that provides abstractions for building applications with large language models, including chains, agents, and memory."
    ),
    Document(
        "d06",
        "Cosine similarity measures the angle between two vectors and is the standard metric for comparing text embeddings in semantic search."
    ),
    Document(
        "d07",
        "RLHF (Reinforcement Learning from Human Feedback) aligns language models with human preferences by training a reward model on human rankings."
    ),
    Document(
        "d08",
        "Chunking strategies for RAG include fixed-size chunks, sentence-aware splits, and recursive character splitting with configurable overlap."
    ),
    Document(
        "d09",
        "The transformer architecture uses self-attention mechanisms to model relationships between all tokens in a sequence simultaneously."
    ),
    Document(
        "d10",
        "Agents use language models as a reasoning engine, enabling them to plan multi-step tasks, call tools, and take actions based on observations."
    ),
]


try:
    store = VectorStore()

    store.add_documents(
        CORPUS
    )

    QUERIES = [
        "How does RAG work?",
        "What algorithms do vector databases use?",
        "How do I split documents for embedding?",
    ]

    for query in QUERIES:

        print(
            f"\nQuery: {query!r}"
        )

        results = store.search(
            query,
            k=3
        )

        for r in results:

            print(
                f" [{r.rank}] "
                f"score={r.score:.4f} | "
                f"{r.document.text[:80]}..."
            )

except Exception as e:
    print("\n[ERROR] Failed to run Semantic Search Pipeline.")
    print("If you are using OpenRouter with an OpenAI model, ensure you have sufficient credits.")
    print(f"Original Error: {str(e)}")