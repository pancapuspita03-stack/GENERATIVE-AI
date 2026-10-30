"""Exercise 4: retrieve ten nodes, LLM-rerank them, and return the best three."""

from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike


def _configure() -> None:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY belum disetel di .env atau environment.")
    model = "nvidia/nemotron-3.5-lightning:free"
    Settings.llm = OpenAILike(
        model=model,
        api_base="https://openrouter.ai/api/v1",
        api_key=api_key,
        is_chat_model=True,
        context_window=32768,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def _score_node(question: str, text: str) -> float:
    prompt = (
        "Score the relevance of the passage to the question from 0 to 10. "
        "Return only one number (an integer or decimal), with no explanation.\n"
        f"Question: {question}\nPassage: {text}"
    )
    raw_score = Settings.llm.complete(prompt).text.strip()
    match = re.search(r"\b(?:10(?:\.0)?|[0-9](?:\.[0-9])?)\b", raw_score)
    if not match:
        raise ValueError(f"LLM mengembalikan skor yang tidak valid: {raw_score!r}")
    score = float(match.group())
    if not 0 <= score <= 10:
        raise ValueError(f"Skor LLM di luar rentang 0-10: {score}")
    return score


def retrieve_and_rerank(
    documents: list[Document], question: str, top_k: int = 10, result_k: int = 3
) -> list[dict[str, Any]]:
    """Use vector retrieval first, then return the highest-scoring three nodes."""
    if not documents:
        raise ValueError("documents tidak boleh kosong.")
    if not question.strip():
        raise ValueError("question tidak boleh kosong.")
    if top_k < 1 or result_k < 1:
        raise ValueError("top_k dan result_k harus lebih besar dari 0.")

    _configure()
    index = VectorStoreIndex.from_documents(documents)
    candidates = VectorIndexRetriever(index=index, similarity_top_k=top_k).retrieve(question)
    scored = [
        {
            "text": node.text,
            "metadata": node.metadata,
            "vector_score": node.score,
            "llm_score": _score_node(question, node.text),
        }
        for node in candidates
    ]
    return sorted(scored, key=lambda item: item["llm_score"], reverse=True)[:result_k]


if __name__ == "__main__":
    sample_documents = [
        Document(text="Vector retrieval finds semantically similar passages."),
        Document(
            text="A reranker can reorder candidates using a more expensive relevance model."
        ),
        Document(text="Metadata describes properties such as topic and publication year."),
        Document(text="A summary index reads every node for synthesis."),
    ]
    for rank, result in enumerate(
        retrieve_and_rerank(sample_documents, "How does reranking improve retrieval?"),
        start=1,
    ):
        print(f"{rank}. score={result['llm_score']:.1f} {result['text']}")
