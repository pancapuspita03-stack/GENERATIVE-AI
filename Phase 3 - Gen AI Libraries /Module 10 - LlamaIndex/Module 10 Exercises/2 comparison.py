"""Exercise 2: compare VectorStoreIndex and SummaryIndex on ten paragraphs."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from llama_index.core import Document, Settings, SummaryIndex, VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


PARAGRAPHS = [
    "RAG retrieves relevant external context before an LLM generates an answer.",
    "Vector databases store embeddings and perform approximate nearest-neighbour search.",
    "Chunk size affects both retrieval precision and the amount of context sent to the LLM.",
    "Hybrid search combines keyword matching with semantic vector similarity.",
    "Metadata filters can restrict retrieval to a topic, year, author, or other field.",
    "A reranker improves a candidate list by scoring each retrieved node for the question.",
    "Summary indexes read all nodes and are useful for questions requiring a document-wide synthesis.",
    "Vector indexes usually retrieve only the most similar nodes and are efficient for specific facts.",
    "Grounded answers should cite the source passages used to produce them.",
    "Evaluation should measure factual retrieval separately from synthesis and completeness.",
]

QUESTIONS = [
    "What does RAG retrieve before generating an answer?",
    "What are the two parts of hybrid search?",
    "Why are metadata filters useful?",
    "Which index is usually more efficient for a specific fact, and why?",
    "What should evaluation measure besides factual retrieval?",
]


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


def compare_indexes() -> dict[str, list[dict[str, str]]]:
    """Run the same five questions against both index types."""
    _configure()
    documents = [Document(text=text, metadata={"paragraph": number})
                 for number, text in enumerate(PARAGRAPHS, start=1)]
    vector_engine = VectorStoreIndex.from_documents(documents).as_query_engine(
        similarity_top_k=2
    )
    summary_engine = SummaryIndex.from_documents(documents).as_query_engine()

    results: dict[str, list[dict[str, str]]] = {"vector": [], "summary": []}
    for question in QUESTIONS:
        results["vector"].append(
            {"question": question, "answer": str(vector_engine.query(question))}
        )
        results["summary"].append(
            {"question": question, "answer": str(summary_engine.query(question))}
        )
    return results


if __name__ == "__main__":
    results = compare_indexes()
    for question_number, question in enumerate(QUESTIONS, start=1):
        print(f"\nQ{question_number}: {question}")
        print("VectorStoreIndex:", results["vector"][question_number - 1]["answer"])
        print("SummaryIndex:", results["summary"][question_number - 1]["answer"])
    print("\nQuality comparison:")
    print("- Specific fact retrieval: VectorStoreIndex is usually more focused and efficient.")
    print("- Document-wide synthesis: SummaryIndex sees all ten paragraphs and is usually more complete.")
