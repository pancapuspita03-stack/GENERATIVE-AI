def embed_texts(texts: list[str]) -> list[list[float]]:
    """Simulate an embedding API call (cached/dummy)."""
    return [[hash(t) % 100 / 100.0, 0.42, 0.87] for t in texts]