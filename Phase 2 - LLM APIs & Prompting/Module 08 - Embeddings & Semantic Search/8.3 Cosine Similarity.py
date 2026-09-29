import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(a, b) / (norm_a * norm_b)
    )


def pairwise_similarity(matrix: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity for all rows in a matrix.
    Returns an (n, n) matrix. All rows must be non-zero.
    """

    # Normalise all rows to unit length first
    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True
    )

    norms = np.where(
        norms == 0,
        1,
        norms
    )

    normed = matrix / norms

    # Dot product of normalised vectors = cosine similarity
    return (
        normed @ normed.T
    ).astype(np.float32)


# Example
rng = np.random.default_rng(42)

vecs = rng.standard_normal(
    (4, 8)
).astype(np.float32)

sim_matrix = pairwise_similarity(vecs)

print("Pairwise similarities:")

for i in range(4):
    for j in range(i + 1, 4):
        print(
            f" vec[{i}] vs vec[{j}]: {sim_matrix[i, j]:.4f}"
        )


# Diagonal is always 1.0 (a vector is identical to itself)
print(
    f"Diagonal: {np.diag(sim_matrix)}"
)