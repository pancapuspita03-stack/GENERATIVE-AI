# 10.5 Retriever - Used Independently (Halaman 119-120)
import asyncio
import os
from typing import Any

import requests
from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.retrievers import VectorIndexRetriever


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


load_dotenv()

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
if not openrouter_api_key:
    raise RuntimeError(
        "OPENROUTER_API_KEY belum disetel di file .env atau environment."
    )
if not openrouter_api_key.startswith("sk-or-v1-"):
    raise RuntimeError(
        "OPENROUTER_API_KEY bukan key OpenRouter yang dikenali. "
        "Gunakan key dari https://openrouter.ai/keys."
    )

Settings.embed_model = OpenRouterNvidiaEmbedding(api_key=openrouter_api_key)

documents = [
    Document(text="ChromaDB is an open-source embedding database with a simple Python API.", metadata={"tag": "vector-db"}),
    Document(text="Pinecone is a managed vector database designed for production workloads.", metadata={"tag": "vector-db"}),
    Document(text="FAISS is a library by Meta for efficient similarity search on dense vectors.", metadata={"tag": "vector-search"}),
    Document(text="pgvector adds vector similarity search to PostgreSQL.", metadata={"tag": "vector-db"}),
    Document(text="Weaviate is an open-source vector database with a GraphQL API.", metadata={"tag": "vector-db"}),
]

index = VectorStoreIndex.from_documents(documents)
retriever = VectorIndexRetriever(index=index, similarity_top_k=3)

if __name__ == "__main__":
    # Retrieve without synthesising an answer
    nodes = retriever.retrieve("open source vector databases")
    print("Retrieved nodes:")
    for node in nodes:
        print(f" [{node.score:.4f}] {node.text}")
        print(f"  tag: {node.metadata.get('tag')}")