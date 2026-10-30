"""Exercise 4: export and import ChromaDB collections as JSON-compatible records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb


def export_collection(collection: Any, output_path: str | Path) -> list[dict[str, Any]]:
    """Export all records, including IDs, text, embeddings, and metadata."""
    total = collection.count()
    if total == 0:
        records: list[dict[str, Any]] = []
    else:
        data = collection.get(include=["documents", "embeddings", "metadatas"])
        records = [
            {
                "id": item_id,
                "text": text,
                "embedding": embedding.tolist()
                if hasattr(embedding, "tolist")
                else list(embedding),
                "metadata": metadata,
            }
            for item_id, text, embedding, metadata in zip(
                data["ids"], data["documents"], data["embeddings"], data["metadatas"]
            )
        ]
    Path(output_path).write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def import_collection(
    client: Any, collection_name: str, input_path: str | Path, batch_size: int = 500
) -> Any:
    """Create/update a collection from the JSON export."""
    if batch_size < 1:
        raise ValueError("batch_size harus lebih besar dari 0.")
    records = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Export JSON harus berupa list.")
    collection = client.get_or_create_collection(collection_name)
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        collection.upsert(
            ids=[record["id"] for record in batch],
            documents=[record["text"] for record in batch],
            embeddings=[record["embedding"] for record in batch],
            metadatas=[record["metadata"] for record in batch],
        )
    return collection


if __name__ == "__main__":
    client = chromadb.Client()
    print("Use export_collection(collection, 'backup.json') and import_collection(client, name, 'backup.json').")
