"""Exercise 2: search multiple ChromaDB collections and merge by score."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import chromadb


def embed(text: str, dimension: int = 32) -> list[float]:
    values = [
        (int.from_bytes(hashlib.sha256(f"{index}:{text}".encode()).digest()[:4], "big") / 2**32)
        * 2
        - 1
        for index in range(dimension)
    ]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


class MultiCollectionSearch:
    def __init__(self, client: Any | None = None) -> None:
        self.client = client or chromadb.Client()

    def search(
        self, query: str, collection_names: list[str], top_k_per_collection: int = 5, limit: int = 10
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query tidak boleh kosong.")
        if not collection_names:
            raise ValueError("collection_names tidak boleh kosong.")
        if top_k_per_collection < 1 or limit < 1:
            raise ValueError("top_k_per_collection dan limit harus lebih besar dari 0.")

        ranked: list[dict[str, Any]] = []
        query_embedding = embed(query)
        for collection_name in collection_names:
            collection = self.client.get_collection(collection_name)
            collection_size = collection.count()
            if collection_size == 0:
                continue
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k_per_collection, collection_size),
                include=["documents", "metadatas", "distances"],
            )
            for document, metadata, distance, item_id in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
                results["ids"][0],
            ):
                ranked.append(
                    {
                        "id": item_id,
                        "text": document,
                        "metadata": metadata,
                        "collection": collection_name,
                        "score": 1.0 - float(distance),
                    }
                )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:limit]


if __name__ == "__main__":
    print("Create collections, then call MultiCollectionSearch().search(...).")
