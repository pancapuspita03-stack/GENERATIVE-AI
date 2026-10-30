from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_index.core import Document, Settings, SummaryIndex, VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterCondition
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike


# ==========================================
# SHARED CONFIGURATION
# ==========================================
def _configure_llama_index() -> None:
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


# ==========================================
# EXERCISE 1: Document Q&A over plain-text files
# ==========================================
def build_index(txt_folder: str | Path) -> VectorStoreIndex:
    """Load every .txt file in *txt_folder* and build a vector index."""
    folder = Path(txt_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder dokumen tidak ditemukan: {folder}")
    if not list(folder.glob("*.txt")):
        raise ValueError(f"Tidak ada file .txt di folder: {folder}")

    _configure_llama_index()
    documents = SimpleDirectoryReader(
        input_dir=str(folder),
        required_exts=[".txt"],
        recursive=False,
    ).load_data()
    return VectorStoreIndex.from_documents(documents)


def ask(question: str, txt_folder: str | Path | None = None) -> dict[str, Any]:
    """Answer *question* and return the answer together with source filenames."""
    if not question.strip():
        raise ValueError("question tidak boleh kosong.")

    folder = Path(txt_folder) if txt_folder is not None else Path(__file__).with_name("documents")
    index = build_index(folder)
    response = index.as_query_engine(similarity_top_k=3).query(question)
    sources = sorted(
        {
            str(node.metadata.get("file_name") or node.metadata.get("file_path"))
            for node in response.source_nodes
            if node.metadata.get("file_name") or node.metadata.get("file_path")
        }
    )
    return {"answer": str(response), "sources": sources}


# ==========================================
# EXERCISE 2: VectorStoreIndex vs SummaryIndex
# ==========================================
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


def compare_indexes() -> dict[str, list[dict[str, str]]]:
    """Run the same five questions against both index types."""
    _configure_llama_index()
    documents = [
        Document(text=text, metadata={"paragraph": number})
        for number, text in enumerate(PARAGRAPHS, start=1)
    ]
    vector_engine = VectorStoreIndex.from_documents(documents).as_query_engine(similarity_top_k=2)
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


# ==========================================
# EXERCISE 3: Retrieve by Topic & Year Metadata
# ==========================================
def build_tagged_index(documents: list[Document]) -> VectorStoreIndex:
    """Build an index whose documents contain ``year`` and ``topic`` metadata."""
    if not documents:
        raise ValueError("documents tidak boleh kosong.")
    _configure_llama_index()
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


# ==========================================
# EXERCISE 4: Retrieve and LLM-Rerank Nodes
# ==========================================
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

    _configure_llama_index()
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


# ==========================================
# MAIN EXECUTION (Runs all 4 exercises)
# ==========================================
if __name__ == "__main__":
    print("=== EXERCISE 1: Document Q&A ===")
    docs_dir = Path("documents")
    if not docs_dir.exists():
        docs_dir.mkdir(exist_ok=True)
        sample_txt = docs_dir / "sample.txt"
        sample_txt.write_text(
            "Dokumen ini membahas tentang implementasi Retrieval-Augmented Generation (RAG) "
            "menggunakan LlamaIndex dan OpenRouter API dengan model LLM open-source."
        )
    try:
        result_ex1 = ask("Apa topik utama dokumen-dokumen ini?")
        print("Answer:", result_ex1["answer"])
        print("Sources:", ", ".join(result_ex1["sources"]))
    except Exception as e:
        print("Exercise 1 skipped/error:", e)

    print("\n=== EXERCISE 2: VectorStoreIndex vs SummaryIndex ===")
    results_ex2 = compare_indexes()
    for question_number, question in enumerate(QUESTIONS, start=1):
        print(f"\nQ{question_number}: {question}")
        print("VectorStoreIndex:", results_ex2["vector"][question_number - 1]["answer"])
        print("SummaryIndex:", results_ex2["summary"][question_number - 1]["answer"])
    print("\nQuality comparison:")
    print("- Specific fact retrieval: VectorStoreIndex is usually more focused and efficient.")
    print("- Document-wide synthesis: SummaryIndex sees all ten paragraphs and is usually more complete.")

    print("\n=== EXERCISE 3: Retrieve by Metadata (Topic & Year) ===")
    tagged_documents = [
        Document(text="Embeddings map text into vectors.", metadata={"year": 2024, "topic": "embeddings"}),
        Document(text="Chunking splits long documents.", metadata={"year": 2024, "topic": "chunking"}),
        Document(text="Older embedding methods used sparse vectors.", metadata={"year": 2023, "topic": "embeddings"}),
    ]
    tagged_index = build_tagged_index(tagged_documents)
    nodes_ex3 = retrieve_by_metadata(
        tagged_index,
        "How do embeddings represent text?",
        {"year": 2024, "topic": "embeddings"},
    )
    for node in nodes_ex3:
        print(node.metadata, "->", node.text)

    print("\n=== EXERCISE 4: Retrieve and LLM-Rerank ===")
    sample_documents_ex4 = [
        Document(text="Vector retrieval finds semantically similar passages."),
        Document(text="A reranker can reorder candidates using a more expensive relevance model."),
        Document(text="Metadata describes properties such as topic and publication year."),
        Document(text="A summary index reads every node for synthesis."),
    ]
    for rank, result in enumerate(
        retrieve_and_rerank(sample_documents_ex4, "How does reranking improve retrieval?"),
        start=1,
    ):
        print(f"{rank}. score={result['llm_score']:.1f} {result['text']}")