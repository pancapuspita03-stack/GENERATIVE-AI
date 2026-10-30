"""Module 11 exercises: ChromaDB benchmark, search, tenant isolation, and migration."""

from __future__ import annotations

import hashlib
import json
import math
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import chromadb


EMBEDDING_DIMENSION = 32


def embed(text: str) -> list[float]:
    """Buat embedding deterministik untuk demo tanpa API eksternal."""
    values = [
        (int.from_bytes(
            hashlib.sha256(f"{index}:{text}".encode()).digest()[:4], "big"
        ) / 2**32) * 2 - 1
        for index in range(EMBEDDING_DIMENSION)
    ]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


# ============================================================
# EXERCISE 1: Benchmark ChromaDB dengan 1.000 dokumen
# ============================================================

@dataclass(frozen=True)
class BenchmarkResult:
    average_query_latency_ms: float
    p95_query_latency_ms: float
    throughput_results_per_second: float
    document_count: int
    query_count: int


def run_benchmark(client: Any) -> BenchmarkResult:
    collection = client.get_or_create_collection(
        name="benchmark",
        metadata={"hnsw:space": "cosine"},
    )

    documents = [
        f"Mock document {index}: vector database benchmark."
        for index in range(1000)
    ]
    collection.upsert(
        ids=[f"doc-{index}" for index in range(1000)],
        documents=documents,
        embeddings=[embed(document) for document in documents],
    )

    queries = [
        f"benchmark query {index} topic "
        f"{random.Random(7 + index).randint(0, 999)}"
        for index in range(100)
    ]

    latencies: list[float] = []
    results_count = 0
    started = time.perf_counter()

    for query in queries:
        query_started = time.perf_counter()
        result = collection.query(
            query_embeddings=[embed(query)],
            n_results=5,
            include=["documents", "distances"],
        )
        results_count += len(result["ids"][0])
        latencies.append((time.perf_counter() - query_started) * 1000)

    elapsed = time.perf_counter() - started
    ordered_latencies = sorted(latencies)
    p95_index = min(
        len(ordered_latencies) - 1,
        math.ceil(len(ordered_latencies) * 0.95) - 1,
    )

    return BenchmarkResult(
        average_query_latency_ms=mean(latencies),
        p95_query_latency_ms=ordered_latencies[p95_index],
        throughput_results_per_second=results_count / elapsed,
        document_count=collection.count(),
        query_count=len(queries),
    )


# ============================================================
# EXERCISE 2: Cari beberapa collection, lalu gabungkan hasilnya
# ============================================================

def search_multiple_collections(
    client: Any,
    query: str,
    collection_names: list[str],
    top_k_per_collection: int = 3,
    limit: int = 5,
) -> list[dict[str, Any]]:
    if not query.strip():
        raise ValueError("Query tidak boleh kosong.")
    if not collection_names:
        raise ValueError("Daftar collection tidak boleh kosong.")
    if top_k_per_collection < 1 or limit < 1:
        raise ValueError("Nilai top_k_per_collection dan limit harus positif.")

    ranked: list[dict[str, Any]] = []
    query_embedding = embed(query)

    for collection_name in collection_names:
        collection = client.get_collection(collection_name)
        count = collection.count()
        if count == 0:
            continue

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k_per_collection, count),
            include=["documents", "metadatas", "distances"],
        )

        for item_id, text, metadata, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            ranked.append(
                {
                    "id": item_id,
                    "text": text,
                    "metadata": metadata,
                    "collection": collection_name,
                    "score": 1.0 - float(distance),
                }
            )

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:limit]


# ============================================================
# EXERCISE 3: Pisahkan data berdasarkan tenant
# ============================================================

class TenantVectorStore:
    def __init__(self, client: Any, collection_prefix: str = "tenant") -> None:
        self.client = client
        self.collection_prefix = collection_prefix

    def _collection(self, tenant_id: str) -> Any:
        if not tenant_id or any(
            character not in "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in tenant_id
        ):
            raise ValueError("tenant_id hanya boleh berisi huruf, angka, _ atau -.")

        return self.client.get_or_create_collection(
            name=f"{self.collection_prefix}_{tenant_id}",
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        tenant_id: str,
        item_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._collection(tenant_id).upsert(
            ids=[item_id],
            documents=[text],
            embeddings=[embed(text)],
            metadatas=[metadata or {}],
        )

    def search(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not query.strip():
            raise ValueError("Query tidak boleh kosong.")
        if top_k < 1:
            raise ValueError("top_k harus lebih besar dari 0.")

        collection = self._collection(tenant_id)
        count = collection.count()
        if count == 0:
            return []

        results = collection.query(
            query_embeddings=[embed(query)],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        return [
            {
                "id": item_id,
                "text": text,
                "metadata": metadata,
                "score": 1.0 - float(distance),
            }
            for item_id, text, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]


# ============================================================
# EXERCISE 4: Ekspor dan impor collection melalui JSON
# ============================================================

def export_collection(collection: Any, output_path: str | Path) -> list[dict[str, Any]]:
    count = collection.count()
    if count == 0:
        records: list[dict[str, Any]] = []
    else:
        data = collection.get(
            include=["documents", "embeddings", "metadatas"]
        )
        records = [
            {
                "id": item_id,
                "text": text,
                "embedding": (
                    embedding.tolist()
                    if hasattr(embedding, "tolist")
                    else list(embedding)
                ),
                "metadata": metadata or {},
            }
            for item_id, text, embedding, metadata in zip(
                data["ids"],
                data["documents"],
                data["embeddings"],
                data["metadatas"],
            )
        ]

    Path(output_path).write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    return records


def import_collection(
    client: Any,
    collection_name: str,
    input_path: str | Path,
    batch_size: int = 500,
) -> Any:
    if batch_size < 1:
        raise ValueError("batch_size harus lebih besar dari 0.")

    records = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("File JSON harus berisi list records.")

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        collection.upsert(
            ids=[record["id"] for record in batch],
            documents=[record["text"] for record in batch],
            embeddings=[record["embedding"] for record in batch],
            metadatas=[record.get("metadata") or {} for record in batch],
        )

    return collection


# ============================================================
# Jalankan empat latihan dan tampilkan empat output
# ============================================================

def main() -> None:
    client = chromadb.EphemeralClient()

    print("=== OUTPUT 1: Benchmark ===")
    print(run_benchmark(client))

    print("\n=== OUTPUT 2: Multi-collection search ===")
    collection_a = client.get_or_create_collection(
        name="search_docs_a",
        metadata={"hnsw:space": "cosine"},
    )
    collection_a.upsert(
        ids=["a1", "a2"],
        documents=[
            "ChromaDB menyimpan dan mencari embedding.",
            "Vector search membantu menemukan dokumen terkait.",
        ],
        embeddings=[
            embed("ChromaDB menyimpan dan mencari embedding."),
            embed("Vector search membantu menemukan dokumen terkait."),
        ],
        metadatas=[{"source": "docs_a"}, {"source": "docs_a"}],
    )

    collection_b = client.get_or_create_collection(
        name="search_docs_b",
        metadata={"hnsw:space": "cosine"},
    )
    collection_b.upsert(
        ids=["b1"],
        documents=["Metadata dapat digunakan untuk memfilter dokumen."],
        embeddings=[embed("Metadata dapat digunakan untuk memfilter dokumen.")],
        metadatas=[{"source": "docs_b"}],
    )

    search_results = search_multiple_collections(
        client,
        query="vector search dan embedding",
        collection_names=["search_docs_a", "search_docs_b"],
        top_k_per_collection=2,
        limit=3,
    )
    for result in search_results:
        print(
            f"{result['collection']} | score={result['score']:.3f} | "
            f"{result['text']}"
        )

    print("\n=== OUTPUT 3: Tenant isolation ===")
    tenant_store = TenantVectorStore(client)
    tenant_store.add(
        "tenant-a",
        "a1",
        "Dokumen rahasia milik tenant A.",
        {"tenant": "tenant-a"},
    )
    tenant_store.add(
        "tenant-b",
        "b1",
        "Dokumen privat milik tenant B.",
        {"tenant": "tenant-b"},
    )

    tenant_a_results = tenant_store.search("tenant-a", "dokumen rahasia")
    tenant_b_results = tenant_store.search("tenant-b", "dokumen rahasia")
    print("Tenant A:", tenant_a_results)
    print("Tenant B:", tenant_b_results)
    assert all(
        result["metadata"]["tenant"] == "tenant-a"
        for result in tenant_a_results
    )
    assert all(
        result["metadata"]["tenant"] == "tenant-b"
        for result in tenant_b_results
    )

    print("\n=== OUTPUT 4: JSON export/import ===")
    source = client.get_or_create_collection(
        name="migration_source",
        metadata={"hnsw:space": "cosine"},
    )
    source.upsert(
        ids=["m1", "m2"],
        documents=[
            "Record pertama untuk migrasi.",
            "Record kedua untuk migrasi.",
        ],
        embeddings=[
            embed("Record pertama untuk migrasi."),
            embed("Record kedua untuk migrasi."),
        ],
        metadatas=[
            {"category": "demo", "number": 1},
            {"category": "demo", "number": 2},
        ],
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        backup_path = Path(temp_dir) / "chroma_backup.json"
        exported = export_collection(source, backup_path)
        restored = import_collection(
            client,
            "migration_restored",
            backup_path,
        )
        print(f"Records diekspor: {len(exported)}")
        print(f"Records diimpor: {restored.count()}")
        print("Contoh hasil:", restored.get(include=["documents", "metadatas"]))

    print("\nSelesai: empat latihan berhasil dijalankan.")


if __name__ == "__main__":
    main()