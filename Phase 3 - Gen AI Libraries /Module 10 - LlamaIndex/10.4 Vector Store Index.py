# 10.4 VectorStoreIndex
import asyncio
import os
from typing import Any

import requests
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.openai_like import OpenAILike
from openai import RateLimitError


class OpenRouterNvidiaEmbedding(BaseEmbedding):
    """Generate embeddings through OpenRouter's NVIDIA embeddings endpoint."""

    api_key: str
    api_base: str = "https://openrouter.ai/api/v1"
    model_name: str = "nvidia/nemotron-3.5-embed-1b:free"

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.api_base.rstrip('/')}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model_name, "input": texts},
            timeout=60,
        )
        response.raise_for_status()

        payload: Any = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError(
                "OpenRouter returned an invalid embedding response: "
                "the number of vectors did not match the input."
            )

        indexed_embeddings: list[tuple[int, list[float]]] = []
        for item in data:
            if not isinstance(item, dict):
                raise RuntimeError("OpenRouter returned a malformed embedding.")
            index = item.get("index")
            embedding = item.get("embedding")
            if (
                not isinstance(index, int)
                or not isinstance(embedding, list)
                or not all(isinstance(value, (int, float)) for value in embedding)
            ):
                raise RuntimeError("OpenRouter returned a malformed embedding.")
            indexed_embeddings.append(
                (index, [float(value) for value in embedding])
            )

        indexed_embeddings.sort(key=lambda result: result[0])
        return [embedding for _, embedding in indexed_embeddings]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_embeddings([text])[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_embeddings([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._get_embeddings(texts)


def main() -> None:
    load_dotenv()

    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY belum disetel di file .env atau environment."
        )
    if not openrouter_api_key.startswith("sk-or-v1-"):
        raise RuntimeError(
            "OPENROUTER_API_KEY bukan key OpenRouter yang dikenali. "
            "Gunakan key dari https://openrouter.ai/keys, bukan OPENAI_API_KEY."
        )

    openrouter_model = os.environ.get(
        "OPENROUTER_MODEL",
        "openrouter/free",
    )
    # OpenRouter uses the OpenAI-compatible protocol; this base URL routes requests to OpenRouter.
    openrouter_llm = OpenAILike(
        model=openrouter_model,
        api_base="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        is_chat_model=True,
        context_window=32768,
        temperature=0.1,
    )
    Settings.llm = openrouter_llm
    Settings.embed_model = OpenRouterNvidiaEmbedding(api_key=openrouter_api_key)

    documents = [
        Document(
            text="RAG combines retrieval with generation to answer questions from external knowledge."
        ),
        Document(
            text="Fine-tuning trains a model on new data to improve performance on a specific task."
        ),
        Document(
            text="Prompt engineering optimises the input to an LLM to get better outputs without any training."
        ),
        Document(
            text="Vector databases store high-dimensional embeddings and support fast approximate nearest-neighbour search."
        ),
        Document(
            text="LangChain provides abstractions for building chains, agents, and memory-enabled LLM applications."
        ),
    ]

    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=3)
    try:
        response = query_engine.query(
            "What is the difference between RAG and fine-tuning?"
        )
    except RateLimitError as error:
        raise RuntimeError(
            "OpenRouter sedang terkena rate limit. Tunggu beberapa saat, "
            "gunakan OPENROUTER_MODEL lain, atau tambahkan provider key di "
            "https://openrouter.ai/settings/integrations."
        ) from error

    print("Answer:", response)
    print("\nSource nodes:")
    for node in response.source_nodes:
        score = node.score if node.score is not None else 0.0
        print(f" Score {score:.4f}: {node.text[:60]}...")


if __name__ == "__main__":
    main()