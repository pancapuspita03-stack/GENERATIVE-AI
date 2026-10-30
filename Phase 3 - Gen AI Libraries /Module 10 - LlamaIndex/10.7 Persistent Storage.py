# 10.7 Persistent Storage (Halaman 122-123)
import asyncio
import os
from typing import Any

import requests
from dotenv import load_dotenv
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.openai_like import OpenAILike


class OpenRouterNvidiaEmbedding(BaseEmbedding):
    api_key: str
    api_base: str = "https://openrouter.ai/api/v1"
    model_name: str = "nvidia/nemotron-3.5-embed-1b:free"

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            f"{self.api_base.rstrip('/')}/embeddings",
            headers={
                "Authorization": "Bearer " + self.api_key,
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
        if [index for index, _ in indexed_embeddings] != list(range(len(texts))):
            raise RuntimeError("OpenRouter returned invalid embedding indexes.")
        return [embedding for _, embedding in indexed_embeddings]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._get_embeddings([text])[0]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._get_embeddings([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._get_embeddings(texts)

PERSIST_DIR = "/tmp/llama_index_store"

documents = [
    Document(text="LlamaIndex persists indexes to disk as JSON files."),
    Document(text="Loading a persisted index avoids re-embedding on every run."),
]

def main() -> None:
    load_dotenv()
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not openrouter_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY belum disetel di file .env atau environment."
        )

    Settings.llm = OpenAILike(
        model=os.environ.get("OPENROUTER_MODEL", "openrouter/free"),
        api_base="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        is_chat_model=True,
        context_window=32768,
        temperature=0.1,
    )
    Settings.embed_model = OpenRouterNvidiaEmbedding(api_key=openrouter_api_key)

    # Build and save index
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Index saved to {PERSIST_DIR}")

    # Load index back - no re-embedding needed
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    loaded_index = load_index_from_storage(storage_context)
    query_engine = loaded_index.as_query_engine()
    response = query_engine.query("How does LlamaIndex handle persistence?")
    print("After reload:", response)


if __name__ == "__main__":
    main()
