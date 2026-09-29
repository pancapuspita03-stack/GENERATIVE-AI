import os
import re
import sqlite3
import hashlib
import numpy as np

from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

# ============================================================
# SETUP
# ============================================================

load_dotenv()

openai_client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class Document:
    id: str
    text: str


# ============================================================
# EMBEDDING HELPER
# ============================================================

def embed_texts(
    texts: list[str],
    model: str = "openai/text-embedding-3-small"
) -> np.ndarray:
    """
    Embed a list of texts using OpenAI embeddings.
    Returns a normalized float32 numpy array.
    """

    response = openai_client.embeddings.create(
        input=texts,
        model=model
    )

    vectors = sorted(
        response.data,
        key=lambda x: x.index
    )

    embeddings = np.array(
        [v.embedding for v in vectors],
        dtype=np.float32
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    norms = np.where(
        norms == 0,
        1,
        norms
    )

    return embeddings / norms


# ============================================================
# EXERCISE 1
# DUPLICATE DETECTOR
# ============================================================

class DuplicateDetector:
    """
    Detect near-duplicate documents using cosine similarity.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        model: str = "openai/text-embedding-3-small"
    ):
        self.threshold = threshold
        self.model = model

    def find_duplicates(
        self,
        documents: list[Document]
    ) -> list[tuple[Document, Document, float]]:

        if len(documents) < 2:
            return []

        texts = [
            doc.text
            for doc in documents
        ]

        embeddings = embed_texts(
            texts,
            model=self.model
        )

        duplicates = []

        # Pairwise cosine similarity
        similarity_matrix = embeddings @ embeddings.T

        for i in range(len(documents)):

            for j in range(i + 1, len(documents)):

                score = float(
                    similarity_matrix[i, j]
                )

                if score >= self.threshold:

                    duplicates.append(
                        (
                            documents[i],
                            documents[j],
                            score
                        )
                    )

        return duplicates


def run_duplicate_detector_demo():

    print("\n" + "=" * 70)
    print("EXERCISE 1 - DUPLICATE DETECTOR")
    print("=" * 70)

    corpus = [
        Document(
            "doc01",
            "Retrieval-Augmented Generation combines document retrieval with language model generation."
        ),

        Document(
            "doc02",
            "RAG combines document retrieval with language model generation."
        ),

        Document(
            "doc03",
            "Vector databases store embeddings and support semantic search."
        ),

        Document(
            "doc04",
            "Vector databases store vector embeddings and enable semantic search."
        ),

        Document(
            "doc05",
            "Python is a popular programming language."
        ),

        Document(
            "doc06",
            "Python is one of the most popular programming languages."
        ),

        Document(
            "doc07",
            "Fine-tuning adapts a pretrained model to a specific task."
        ),

        Document(
            "doc08",
            "Fine-tuning adapts a pre-trained model to a particular task."
        ),

        Document(
            "doc09",
            "The transformer architecture uses self-attention."
        ),

        Document(
            "doc10",
            "Transformers use self-attention mechanisms."
        ),

        Document(
            "doc11",
            "Prompt engineering designs instructions for language models."
        ),

        Document(
            "doc12",
            "Prompt engineering focuses on designing instructions for LLMs."
        ),

        Document(
            "doc13",
            "Embeddings represent text as numerical vectors."
        ),

        Document(
            "doc14",
            "Text embeddings represent text using numerical vectors."
        ),

        Document(
            "doc15",
            "Cosine similarity measures the angle between two vectors."
        ),

        Document(
            "doc16",
            "Cosine similarity measures the angle between vectors."
        ),

        Document(
            "doc17",
            "Agents can use tools to complete multi-step tasks."
        ),

        Document(
            "doc18",
            "AI agents can call tools to perform multi-step tasks."
        ),

        Document(
            "doc19",
            "Chunking splits documents into smaller pieces."
        ),

        Document(
            "doc20",
            "Document chunking divides documents into smaller pieces."
        ),

        Document(
            "doc21",
            "LLMs predict the next token in a sequence."
        ),

        Document(
            "doc22",
            "Large language models predict the next token in a sequence."
        ),

        Document(
            "doc23",
            "RLHF aligns models with human preferences."
        ),

        Document(
            "doc24",
            "RLHF helps align language models with human preferences."
        ),

        Document(
            "doc25",
            "A vector store indexes embeddings for retrieval."
        ),

        Document(
            "doc26",
            "Vector stores index embeddings to support retrieval."
        ),

        Document(
            "doc27",
            "Semantic search retrieves documents based on meaning."
        ),

        Document(
            "doc28",
            "Semantic search finds documents based on their meaning."
        ),

        Document(
            "doc29",
            "BM25 is a keyword-based information retrieval algorithm."
        ),

        Document(
            "doc30",
            "BM25 is an information retrieval algorithm based on keywords."
        ),

        Document(
            "doc31",
            "OpenAI provides APIs for large language models."
        ),

        Document(
            "doc32",
            "OpenAI provides APIs that allow applications to use language models."
        ),

        Document(
            "doc33",
            "Claude is an AI assistant developed by Anthropic."
        ),

        Document(
            "doc34",
            "Claude is an AI assistant created by Anthropic."
        ),

        Document(
            "doc35",
            "GPT models can process natural language."
        ),

        Document(
            "doc36",
            "GPT models are capable of processing natural language."
        ),

        Document(
            "doc37",
            "RAG can reduce hallucinations by retrieving external information."
        ),

        Document(
            "doc38",
            "RAG may reduce hallucinations by retrieving external information."
        ),

        Document(
            "doc39",
            "A context window limits how many tokens a model can process."
        ),

        Document(
            "doc40",
            "The context window limits the number of tokens a model can process."
        ),

        Document(
            "doc41",
            "Temperature controls randomness in model generation."
        ),

        Document(
            "doc42",
            "Temperature controls the randomness of generated responses."
        ),

        Document(
            "doc43",
            "Fine-tuning requires training data specific to the target task."
        ),

        Document(
            "doc44",
            "Fine-tuning requires task-specific training data."
        ),

        Document(
            "doc45",
            "Embeddings can be used for semantic similarity."
        ),

        Document(
            "doc46",
            "Embeddings are useful for measuring semantic similarity."
        ),

        Document(
            "doc47",
            "Vector search compares query and document embeddings."
        ),

        Document(
            "doc48",
            "Vector search compares embeddings from queries and documents."
        ),

        Document(
            "doc49",
            "RAG retrieves relevant context before generating an answer."
        ),

        Document(
            "doc50",
            "RAG retrieves relevant context before producing an answer."
        ),
    ]

    detector = DuplicateDetector(
        threshold=0.95
    )

    duplicates = detector.find_duplicates(
        corpus
    )

    print(
        f"\nCorpus size: {len(corpus)} documents"
    )

    print(
        f"Similarity threshold: {detector.threshold}"
    )

    print(
        f"Duplicate pairs found: {len(duplicates)}"
    )

    for doc1, doc2, score in duplicates:

        print(
            f"\n{doc1.id} <-> {doc2.id}"
        )

        print(
            f"Similarity: {score:.4f}"
        )


# ============================================================
# EXERCISE 2
# HYBRID SEARCH
# ============================================================

class HybridSearch:
    """
    Combines semantic similarity with a BM25-style
    keyword score.

    final_score =
        alpha * semantic
        + (1 - alpha) * keyword
    """

    def __init__(
        self,
        documents: list[Document],
        alpha: float = 0.5
    ):
        self.documents = documents
        self.alpha = alpha

        self.embeddings = embed_texts(
            [doc.text for doc in documents]
        )

        self.tokenized_docs = [
            self.tokenize(doc.text)
            for doc in documents
        ]

        self.avg_doc_length = (
            sum(
                len(tokens)
                for tokens in self.tokenized_docs
            )
            / len(self.tokenized_docs)
        )

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(
            r"\b\w+\b",
            text.lower()
        )

    def bm25_score(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
        k1: float = 1.5,
        b: float = 0.75
    ) -> float:

        score = 0.0

        doc_length = len(
            doc_tokens
        )

        unique_terms = set(
            query_tokens
        )

        for term in unique_terms:

            term_frequency = doc_tokens.count(
                term
            )

            if term_frequency == 0:
                continue

            document_frequency = sum(
                1
                for tokens in self.tokenized_docs
                if term in tokens
            )

            n = len(
                self.documents
            )

            idf = np.log(
                1 + (
                    (n - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
            )

            numerator = (
                term_frequency * (k1 + 1)
            )

            denominator = (
                term_frequency
                + k1
                * (
                    1
                    - b
                    + b
                    * (
                        doc_length
                        / self.avg_doc_length
                    )
                )
            )

            score += idf * (
                numerator / denominator
            )

        return float(score)

    def search(
        self,
        query: str,
        k: int = 5
    ) -> list[tuple[Document, float, float, float]]:

        query_embedding = embed_texts(
            [query]
        )[0]

        semantic_scores = (
            self.embeddings
            @ query_embedding
        )

        query_tokens = self.tokenize(
            query
        )

        keyword_scores = np.array([
            self.bm25_score(
                query_tokens,
                doc_tokens
            )
            for doc_tokens in self.tokenized_docs
        ])

        # Normalize BM25 scores
        if keyword_scores.max() > 0:

            keyword_normalized = (
                keyword_scores
                / keyword_scores.max()
            )

        else:

            keyword_normalized = (
                keyword_scores
            )

        final_scores = (
            self.alpha * semantic_scores
            + (1 - self.alpha)
            * keyword_normalized
        )

        top_idx = np.argsort(
            final_scores
        )[::-1][:k]

        return [
            (
                self.documents[i],
                float(semantic_scores[i]),
                float(keyword_normalized[i]),
                float(final_scores[i])
            )
            for i in top_idx
        ]


def run_hybrid_search_demo():

    print("\n" + "=" * 70)
    print("EXERCISE 2 - HYBRID SEARCH")
    print("=" * 70)

    documents = [
        Document(
            "d01",
            "RAG combines retrieval with language model generation."
        ),

        Document(
            "d02",
            "Vector databases store embeddings for semantic search."
        ),

        Document(
            "d03",
            "BM25 is a keyword-based retrieval algorithm."
        ),

        Document(
            "d04",
            "Fine-tuning adapts language models using training data."
        ),

        Document(
            "d05",
            "Semantic search finds documents using embedding similarity."
        ),
    ]

    hybrid = HybridSearch(
        documents,
        alpha=0.7
    )

    query = "How does RAG retrieve documents?"

    print(
        f"\nQuery: {query}"
    )

    print(
        f"Alpha: {hybrid.alpha}"
    )

    results = hybrid.search(
        query,
        k=3
    )

    for doc, semantic, keyword, final in results:

        print(
            f"\n{doc.id}: {doc.text}"
        )

        print(
            f"Semantic score: {semantic:.4f}"
        )

        print(
            f"Keyword score: {keyword:.4f}"
        )

        print(
            f"Final score: {final:.4f}"
        )


# ============================================================
# EXERCISE 3
# VECTOR STORE DELETE AND UPDATE
# ============================================================

class VectorStore:
    """
    In-memory vector store supporting:
    - add
    - search
    - delete
    - update
    """

    def __init__(
        self,
        model: str = "openai/text-embedding-3-small"
    ):
        self.model = model
        self.documents: list[Document] = []
        self.matrix: Optional[np.ndarray] = None

    def rebuild_matrix(self):

        if not self.documents:

            self.matrix = None

            return

        embeddings = embed_texts(
            [
                doc.text
                for doc in self.documents
            ],
            model=self.model
        )

        for doc, embedding in zip(
            self.documents,
            embeddings
        ):
            doc.embedding = embedding

        self.matrix = embeddings

    def add(
        self,
        documents: list[Document]
    ):

        self.documents.extend(
            documents
        )

        self.rebuild_matrix()

    def delete(
        self,
        doc_id: str
    ) -> bool:
        """Delete a document by ID."""

        original_count = len(
            self.documents
        )

        self.documents = [
            doc
            for doc in self.documents
            if doc.id != doc_id
        ]

        deleted = (
            len(self.documents)
            < original_count
        )

        if deleted:
            self.rebuild_matrix()

        return deleted

    def update(
        self,
        doc_id: str,
        new_text: str
    ) -> bool:
        """Update document text and rebuild its embedding."""

        for doc in self.documents:

            if doc.id == doc_id:

                doc.text = new_text

                self.rebuild_matrix()

                return True

        return False

    def search(
        self,
        query: str,
        k: int = 5
    ):

        if not self.documents:

            return []

        if self.matrix is None:

            self.rebuild_matrix()

        query_embedding = embed_texts(
            [query],
            model=self.model
        )[0]

        scores = (
            self.matrix
            @ query_embedding
        )

        k = min(
            k,
            len(self.documents)
        )

        top_idx = np.argsort(
            scores
        )[::-1][:k]

        return [
            (
                self.documents[i],
                float(scores[i])
            )
            for i in top_idx
        ]

    @property
    def size(self):
        return len(
            self.documents
        )


def run_vector_store_demo():

    print("\n" + "=" * 70)
    print("EXERCISE 3 - VECTOR STORE DELETE / UPDATE")
    print("=" * 70)

    store = VectorStore()

    documents = [
        Document(
            "d01",
            "RAG combines retrieval and generation."
        ),

        Document(
            "d02",
            "Vector databases store embeddings."
        ),

        Document(
            "d03",
            "Fine-tuning adapts language models."
        ),
    ]

    store.add(
        documents
    )

    print(
        f"\nInitial size: {store.size}"
    )

    # Delete document
    deleted = store.delete(
        "d02"
    )

    print(
        f"Delete d02: {deleted}"
    )

    print(
        f"Size after delete: {store.size}"
    )

    # Update document
    updated = store.update(
        "d03",
        "Fine-tuning trains a model on task-specific data."
    )

    print(
        f"Update d03: {updated}"
    )

    print(
        f"Size after update: {store.size}"
    )

    print(
        "\nSearch after modifications:"
    )

    results = store.search(
        "How does fine-tuning work?",
        k=2
    )

    for doc, score in results:

        print(
            f"[{score:.4f}] "
            f"{doc.id}: {doc.text}"
        )


# ============================================================
# EXERCISE 4
# EMBEDDING CACHE USING SQLITE
# ============================================================

CACHE_DB = "embedding_cache.db"


def init_cache(
    db_path: str = CACHE_DB
):
    """Create the SQLite cache table if it does not exist."""

    connection = sqlite3.connect(
        db_path
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS embeddings (
            cache_key TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            model TEXT NOT NULL,
            embedding BLOB NOT NULL,
            dimension INTEGER NOT NULL
        )
        """
    )

    connection.commit()

    connection.close()


def make_cache_key(
    text: str,
    model: str
) -> str:
    """
    Create SHA-256 key from text + model.
    """

    value = (
        model
        + ":"
        + text
    )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def embed_with_cache(
    texts: list[str],
    model: str = "openai/text-embedding-3-small",
    db_path: str = CACHE_DB
) -> np.ndarray:
    """
    Embed texts using a SQLite cache.

    Cached texts do not trigger another API request.
    """

    init_cache(
        db_path
    )

    connection = sqlite3.connect(
        db_path
    )

    cursor = connection.cursor()

    results = [None] * len(
        texts
    )

    missing_texts = []
    missing_indices = []

    # --------------------------------------------------------
    # Check cache
    # --------------------------------------------------------

    for index, text in enumerate(
        texts
    ):

        cache_key = make_cache_key(
            text,
            model
        )

        cursor.execute(
            """
            SELECT embedding, dimension
            FROM embeddings
            WHERE cache_key = ?
            """,
            (cache_key,)
        )

        row = cursor.fetchone()

        if row is not None:

            embedding_blob, dimension = row

            vector = np.frombuffer(
                embedding_blob,
                dtype=np.float32
            ).copy()

            results[index] = vector

        else:

            missing_texts.append(
                text
            )

            missing_indices.append(
                index
            )

    # --------------------------------------------------------
    # API call only for missing texts
    # --------------------------------------------------------

    if missing_texts:

        print(
            f"Calling OpenRouter API for "
            f"{len(missing_texts)} new text(s)..."
        )

        response = openai_client.embeddings.create(
            input=missing_texts,
            model=model
        )

        vectors = sorted(
            response.data,
            key=lambda x: x.index
        )

        for index, text, item in zip(
            missing_indices,
            missing_texts,
            vectors
        ):

            vector = np.array(
                item.embedding,
                dtype=np.float32
            )

            results[index] = vector

            cache_key = make_cache_key(
                text,
                model
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO embeddings
                (cache_key, text, model, embedding, dimension)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    cache_key,
                    text,
                    model,
                    vector.tobytes(),
                    len(vector)
                )
            )

    connection.commit()

    connection.close()

    return np.array(
        results,
        dtype=np.float32
    )


def run_embedding_cache_demo():

    print("\n" + "=" * 70)
    print("EXERCISE 4 - SQLITE EMBEDDING CACHE")
    print("=" * 70)

    texts = [
        "What is RAG?",
        "Explain vector databases.",
        "What is RAG?",
    ]

    print("\nFirst call:")

    embeddings_1 = embed_with_cache(
        texts
    )

    print(
        f"Shape: {embeddings_1.shape}"
    )

    print("\nSecond call:")

    embeddings_2 = embed_with_cache(
        texts
    )

    print(
        f"Shape: {embeddings_2.shape}"
    )

    print(
        "\nCache database:"
    )

    print(
        CACHE_DB
    )

    print(
        "\nEmbeddings identical:"
    )

    print(
        np.allclose(
            embeddings_1,
            embeddings_2
        )
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MODULE 08 - EXERCISES 8.8")
    print("=" * 70)

    try:
        # Exercise 1
        run_duplicate_detector_demo()

        # Exercise 2
        run_hybrid_search_demo()

        # Exercise 3
        run_vector_store_demo()

        # Exercise 4
        run_embedding_cache_demo()

        print("\n" + "=" * 70)
        print("ALL MODULE 08 EXERCISES COMPLETED")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        print("Ensure your OPENROUTER_API_KEY is valid and has sufficient credits.")