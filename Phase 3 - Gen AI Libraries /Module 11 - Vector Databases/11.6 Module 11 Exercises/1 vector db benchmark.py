"""Exercise 1: benchmark ChromaDB with 1,000 deterministic mock documents."""

from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import dataclass
from statistics import mean
from typing import Any

import chromadb


EMBEDDING_DIMENSION = 32


def embed(text: str) -> list[float]:
    """Create a deterministic normalized vector without a network API."""
    values = []
    for index in range(EMBEDDING_DIMENSION):
        digest = hashlib.sha256(f"{index}:{text}".encode()).digest()
        values.append((int.from_bytes(digest[:4], "big") / 2**32) * 2 - 1)
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


@dataclass(frozen=True)
class BenchmarkResult:
    average_query_latency_ms: float
    p95_query_latency_ms: float
    throughput_results_per_second: float
    document_count: int
    query_count: int


class VectorDBBenchmark:
    def __init__(self, client: Any | None = None, collection_name: str = "benchmark") -> None:
        self.client = client or chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def insert_mock_documents(self, count: int = 1000) -> None:
        if count < 1:
            raise ValueError("count harus lebih besar dari 0.")
        documents = [f"Mock document {index}: vector database benchmark." for index in range(count)]
        self.collection.upsert(
            ids=[f"doc-{index}" for index in range(count)],
            documents=documents,
            embeddings=[embed(document) for document in documents],
        )

    def run(self, query_count: int = 100, top_k: int = 5, seed: int = 7) -> BenchmarkResult:
        if query_count < 1 or top_k < 1:
            raise ValueError("query_count dan top_k harus lebih besar dari 0.")
        if self.collection.count() == 0:
            self.insert_mock_documents()

        queries = [
            f"benchmark query {index} topic {random.Random(seed + index).randint(0, 999)}"
            for index in range(query_count)
        ]
        latencies: list[float] = []
        results_count = 0
        started = time.perf_counter()
        for query in queries:
            query_started = time.perf_counter()
            result = self.collection.query(
                query_embeddings=[embed(query)],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "distances"],
            )
            results_count += len(result["ids"][0])
            latencies.append((time.perf_counter() - query_started) * 1000)
        elapsed = time.perf_counter() - started
        ordered = sorted(latencies)
        p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
        return BenchmarkResult(
            average_query_latency_ms=mean(latencies),
            p95_query_latency_ms=ordered[p95_index],
            throughput_results_per_second=results_count / elapsed,
            document_count=self.collection.count(),
            query_count=query_count,
        )


if __name__ == "__main__":
    result = VectorDBBenchmark().run()
    print(result)
