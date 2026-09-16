import numpy as np

# Creating arrays
scores = np.array([0.91, 0.76, 0.88, 0.65, 0.95], dtype=np.float32)
print(scores.dtype, scores.shape)  # float32 (5,)

# Zeros, ones, ranges
zeros = np.zeros((3, 4))  # 3 rows, 4 cols of 0.0
rng_vals = np.arange(0, 1.0, 0.1)  # [0.0, 0.1, ..., 0.9]
linspace = np.linspace(0, 1, 5)  # [0.0, 0.25, 0.5, 0.75, 1.0]

# Random - use a seeded Generator for reproducibility
rng = np.random.default_rng(seed=42)
mock_embedding = rng.standard_normal(1536)  # 1536-dim like text-embedding-3-small
print(
    f"Embedding shape:{mock_embedding.shape}, mean:{mock_embedding.mean():.4f}"
)