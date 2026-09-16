import pandas as pd

evals = pd.DataFrame(
    {
        "model": [
            "claude",
            "gpt-4o",
            "claude",
            "gpt-4o",
            "claude",
            "gpt-4o",
        ],
        "task": ["qa", "qa", "summarise", "summarise", "code", "code"],
        "score": [0.91, 0.88, 0.85, 0.82, 0.93, 0.90],
        "latency_ms": [420, 380, 610, 550, 340, 300],
    }
)

# Average score per model
print(evals.groupby("model")["score"].mean())
print()

# Multiple aggregations
summary = evals.groupby("model").agg(
    avg_score=("score", "mean"),
    avg_latency=("latency_ms", "mean"),
    num_tasks=("task", "count"),
)
print(summary)
print()

# Pivot table - model vs task
pivot = evals.pivot_table(
    values="score", index="model", columns="task", aggfunc="mean"
)
print(pivot)