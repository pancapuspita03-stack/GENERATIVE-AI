from dataclasses import dataclass, field
import pathlib
import numpy as np
import pandas as pd


@dataclass
class EvalResult:
    prompt: str
    model: str
    response: str
    latency_ms: float
    score: float = field(default=0.0)


def mock_llm_call(prompt: str, model: str) -> tuple[str, float]:
    """Simulate an LLM call. Returns (response, latency_ms)."""
    rng = np.random.default_rng(abs(hash(prompt + model)) % 2**31)
    latency = rng.uniform(300, 700)
    response = f"[{model}] Answer to:{prompt[:30]}"
    return response, latency


def score_response(
    response: str, expected_keywords: list[str]
) -> float:
    """Simple keyword-based scorer (0.0 – 1.0)."""
    found = sum(
        1 for kw in expected_keywords if kw.lower() in response.lower()
    )
    return found / len(expected_keywords) if expected_keywords else 0.0


# Run evaluation
prompts = [
    ("What is a transformer?", ["attention", "model"]),
    ("Define RAG", ["retrieval", "generation"]),
    ("What is fine-tuning?", ["training", "weights"]),
]
models = ["claude-sonnet-4-5", "gpt-4o"]
results: list[EvalResult] = []

for prompt, keywords in prompts:
    for model in models:
        response, latency = mock_llm_call(prompt, model)
        score = score_response(response, keywords)
        results.append(EvalResult(prompt, model, response, latency, score))

# Analyse with Pandas
df = pd.DataFrame([vars(r) for r in results])
summary = (
    df.groupby("model")
    .agg(
        avg_score=("score", "mean"),
        avg_latency=("latency_ms", "mean"),
    )
    .round(3)
)
print(summary)

# Save results safely (menggunakan pathlib agar kompatibel di Windows/Linux)
output_dir = pathlib.Path("data")
output_dir.mkdir(exist_ok=True)
csv_path = output_dir / "eval_results.csv"

df.to_csv(csv_path, index=False)
print(f"Saved to {csv_path}")