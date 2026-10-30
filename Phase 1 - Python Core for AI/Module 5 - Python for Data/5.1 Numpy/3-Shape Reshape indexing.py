import numpy as np
# Simulate 4 document embeddings of dimension 8
rng = np.random.default_rng(42)
embeddings = rng.standard_normal((4, 8))
print("Shape:", embeddings.shape) # (4, 8)
print("First embedding:", embeddings[0])
print("First 3 dims of all docs:\n", embeddings[:, :3])
# Reshape
flat = embeddings.flatten() # (32,)
back = flat.reshape(4, 8) # (4, 8)
# Boolean indexing
similarity_scores = np.array([0.91, 0.43, 0.78, 0.55])
above_threshold = embeddings[similarity_scores > 0.7]
print(f"Docs above 0.7 similarity:{above_threshold.shape
[0]}")