"""Exercise 3: route tenant operations to isolated ChromaDB collections."""

from __future__ import annotations

import hashlib
import math
import re
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


class TenantVectorStore:
    def __init__(self, client: Any | None = None, collection_prefix: str = "tenant") -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", collection_prefix):
            raise ValueError(
                "collection_prefix hanya boleh berisi huruf, angka, underscore, atau hyphen."
            )
        self.client = client or chromadb.Client()
        self.collection_prefix = collection_prefix

    def _collection(self, tenant_id: str) -> Any:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", tenant_id):
            raise ValueError("tenant_id hanya boleh berisi huruf, angka, underscore, atau hyphen.")
        return self.client.get_or_create_collection(
            name=f"{self.collection_prefix}_{tenant_id}",
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self, tenant_id: str, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]] | None = None
    ) -> None:
        if not ids or len(ids) != len(texts):
            raise ValueError("ids dan texts harus sama-sama berisi dan berukuran sama.")
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas harus berukuran sama dengan texts.")
        collection = self._collection(tenant_id)
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=[embed(text) for text in texts],
            metadatas=metadatas,
        )

    def search(self, tenant_id: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("query tidak boleh kosong.")
        if top_k < 1:
            raise ValueError("top_k harus lebih besar dari 0.")
        collection = self._collection(tenant_id)
        if collection.count() == 0:
            return []
        results = collection.query(
            query_embeddings=[embed(query)],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"id": item_id, "text": text, "metadata": metadata, "score": 1.0 - float(distance)}
            for item_id, text, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]


if __name__ == "__main__":
    store = TenantVectorStore()
    store.add("tenant-a", ["a1"], ["Private document for tenant A"])
    print(store.search("tenant-a", "private document"))
    print(store.search("tenant-b", "private document"))
