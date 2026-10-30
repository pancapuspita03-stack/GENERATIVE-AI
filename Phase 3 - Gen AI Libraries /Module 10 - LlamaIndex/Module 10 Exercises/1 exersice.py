"""Exercise 1: document Q&A over a folder of plain-text files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


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


def ask(
    question: str, txt_folder: str | Path | None = None
) -> dict[str, Any]:
    """Answer *question* and return the answer together with source filenames."""
    if not question.strip():
        raise ValueError("question tidak boleh kosong.")

    folder = Path(txt_folder) if txt_folder is not None else Path(__file__).with_name(
        "documents"
    )
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


if __name__ == "__main__":
    result = ask(
        "Apa topik utama dokumen-dokumen ini?",
    )
    print("Answer:", result["answer"])
    print("Sources:", ", ".join(result["sources"]))