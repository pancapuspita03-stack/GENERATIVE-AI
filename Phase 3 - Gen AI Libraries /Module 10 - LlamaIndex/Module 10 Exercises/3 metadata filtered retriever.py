"""Exercise 3: retrieve only nodes matching both topic and year metadata."""

from __future__ import annotations

from typing import Any

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterCondition
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def build_tagged_index(documents: list[Document]) -> VectorStoreIndex:
    """Build an index whose documents contain ``year`` and ``topic`` metadata."""
    if not documents:
        raise ValueError("documents tidak boleh kosong.")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return VectorStoreIndex.from_documents(documents)


def retrieve_by_metadata(
    index: VectorStoreIndex,
    question: str,
    query_metadata: dict[str, Any],
    top_k: int = 3,
) -> list[Any]:
    """Return nodes matching both ``topic`` and ``year`` from query metadata."""
    if not question.strip():
        raise ValueError("question tidak boleh kosong.")
    if "topic" not in query_metadata or "year" not in query_metadata:
        raise ValueError("query_metadata wajib memiliki topic dan year.")
    if top_k < 1:
        raise ValueError("top_k harus lebih besar dari 0.")

    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="topic", value=query_metadata["topic"]),
            MetadataFilter(key="year", value=query_metadata["year"]),
        ],
        condition=FilterCondition.AND,
    )
    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)
    return retriever.retrieve(question)


if __name__ == "__main__":
    tagged_documents = [
        Document(
            text="Embeddings map text into vectors.",
            metadata={"year": 2024, "topic": "embeddings"},
        ),
        Document(
            text="Chunking splits long documents.",
            metadata={"year": 2024, "topic": "chunking"},
        ),
        Document(
            text="Older embedding methods used sparse vectors.",
            metadata={"year": 2023, "topic": "embeddings"},
        ),
    ]
    tagged_index = build_tagged_index(tagged_documents)
    nodes = retrieve_by_metadata(
        tagged_index,
        "How do embeddings represent text?",
        {"year": 2024, "topic": "embeddings"},
    )
    for node in nodes:
        print(node.metadata, "->", node.text)
