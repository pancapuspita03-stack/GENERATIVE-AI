import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def top_k_similar(
    query: np.ndarray,
    corpus: np.ndarray,
    k: int = 3,
) -> list[tuple[int, float]]:
    """Return indices and scores of the k most similar vectors."""
    # Normalise corpus rows
    norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = corpus / norms

    # Normalise query
    q_norm = np.linalg.norm(query)
    q_normed = query / (q_norm if q_norm > 0 else 1)

    # Matrix–vector multiply -> cosine similarities
    sims = normed @ q_normed
    top_idx = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in top_idx]


# Uji coba
rng = np.random.default_rng(42)
corpus = rng.standard_normal((10, 8))
query = rng.standard_normal(8)
results = top_k_similar(query, corpus, k=3)

for idx, score in results:
    print(f"Doc {idx}: similarity = {score:.4f}")