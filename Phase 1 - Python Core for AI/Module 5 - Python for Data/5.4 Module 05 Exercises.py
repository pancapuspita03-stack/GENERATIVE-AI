import pathlib
import re
import numpy as np
import pandas as pd

# ==========================================
# 1. LLM Benchmark Scores Analysis
# ==========================================


def analyze_benchmark_csv() -> None:
    # 1a. Generate synthetic dataset (>= 20 rows & 4 models)
    models = ["gpt-4o", "claude-sonnet-4-5", "gemini-1.5-pro", "llama-3.1-70b"]
    tasks = ["qa", "summarise", "code", "reasoning", "translation"]

    rng = np.random.default_rng(42)
    rows = []
    for _ in range(24):
        rows.append(
            {
                "model": rng.choice(models),
                "task": rng.choice(tasks),
                "score": round(rng.uniform(0.65, 0.98), 3),
                "latency_ms": rng.integers(250, 800),
            }
        )

    df = pd.DataFrame(rows)

    # 1b. Mean score per model
    mean_scores = df.groupby("model")["score"].mean()

    # 1c. Best-performing task per model
    task_perf = df.groupby(["model", "task"])["score"].mean().reset_index()
    best_tasks = task_perf.loc[task_perf.groupby("model")["score"].idxmax()]

    # 1d. Correlation between score and latency
    corr = df["score"].corr(df["latency_ms"])

    print("--- 1. LLM Benchmark Analysis ---")
    print("\n[Mean Score per Model]:")
    print(mean_scores.to_string())
    print("\n[Best-performing Task per Model]:")
    print(best_tasks.to_string(index=False))
    print(f"\n[Score vs Latency Correlation]: {corr:.4f}\n")


# ==========================================
# 2. L2 Normalisation Function
# ==========================================


def normalise_embeddings(matrix: np.ndarray) -> np.ndarray:
    """L2-normalises each row of a matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def verify_l2_normalisation() -> None:
    rng = np.random.default_rng(42)
    raw_matrix = rng.standard_normal((5, 128))
    normed_matrix = normalise_embeddings(raw_matrix)

    # Verify that row norms equal 1.0
    row_norms = np.linalg.norm(normed_matrix, axis=1)
    is_all_ones = np.allclose(row_norms, 1.0)

    print("--- 2. L2 Normalisation ---")
    print(f"Row Norms: {np.round(row_norms, 4)}")
    print(f"All norms equal 1.0? {is_all_ones}\n")


# ==========================================
# 3. Text Files Statistics Analyzer
# ==========================================


def analyze_text_folder(folder_path: str) -> pd.DataFrame:
    """Reads a folder of .txt files and returns a DataFrame sorted by word_count descending."""
    path = pathlib.Path(folder_path)
    records = []

    for file in path.glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        # Regex split for simple sentence counting (. ! ?)
        sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
        sentence_count = len(sentences)

        records.append(
            {
                "filename": file.name,
                "char_count": char_count,
                "word_count": word_count,
                "sentence_count": sentence_count,
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(by="word_count", ascending=False).reset_index(
            drop=True
        )
    return df


def test_text_analyzer() -> None:
    # Setup temporary dummy text files
    test_dir = pathlib.Path("temp_txt_test")
    test_dir.mkdir(exist_ok=True)

    (test_dir / "doc1.txt").write_text(
        "Python for AI is awesome. Pandas makes analysis easy.", encoding="utf-8"
    )
    (test_dir / "doc2.txt").write_text(
        "Machine learning models require good data preparation.", encoding="utf-8"
    )
    (test_dir / "doc3.txt").write_text(
        "NumPy provides fast vector operations for embeddings.", encoding="utf-8"
    )

    df_result = analyze_text_folder(str(test_dir))

    print("--- 3. Text Files Analyzer ---")
    print(df_result.to_string(index=False))
    print()


# ==========================================
# 4. Pairwise Cosine Similarity Matrix
# ==========================================


def mock_string_embedding(text: str, dim: int = 64) -> np.ndarray:
    """Generate mock embedding vector from a string using hash seed."""
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim)


def compute_pairwise_similarity() -> None:
    corpus = [
        "Retrieval Augmented Generation enhances LLM responses.",
        "Vector databases store embeddings for fast search.",
        "Prompt engineering helps guide language model outputs.",
        "RAG combines document search with generation models.",
        "Fine-tuning adjusts pre-trained model parameters.",
    ]

    # Generate and L2-normalise embeddings
    raw_embeddings = np.array([mock_string_embedding(txt) for txt in corpus])
    normed_embeddings = normalise_embeddings(raw_embeddings)

    # Cosine similarity matrix (Matrix multiplication of L2-normed vectors)
    sim_matrix = normed_embeddings @ normed_embeddings.T

    # Find highest pairwise similarity (excluding self-similarity on diagonal)
    n = len(corpus)
    max_sim = -1.0
    best_pair = (0, 0)

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > max_sim:
                max_sim = sim_matrix[i, j]
                best_pair = (i, j)

    print("--- 4. Pairwise Cosine Similarity ---")
    print("Cosine Similarity Matrix Shape:", sim_matrix.shape)
    print(
        f"\nHighest Similarity Pair ({max_sim:.4f}):\n"
        f" String A [{best_pair[0]}]: '{corpus[best_pair[0]]}'\n"
        f" String B [{best_pair[1]}]: '{corpus[best_pair[1]]}'"
    )


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    analyze_benchmark_csv()
    verify_l2_normalisation()
    test_text_analyzer()
    compute_pairwise_similarity()