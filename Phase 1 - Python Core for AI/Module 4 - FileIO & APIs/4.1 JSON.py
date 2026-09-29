import json
import pathlib

# Save model evaluation results
results = {
    "model": "claude-sonnet-4-5",
    "benchmark": "MMLU",
    "scores": {"science": 0.91, "math": 0.88, "history": 0.85},
    "total_samples": 14042,
    "timestamp": "2025-01-15T09:30:00Z",
}
out = pathlib.Path("results.json")

out.write_text(json.dumps(results, indent=2), encoding="utf-8")

# Read back and use
data = json.loads(out.read_text(encoding="utf-8"))
print(f"Model:{data['model']}")
avg = sum(data["scores"].values()) / len(data["scores"])
print(f"Average score:{avg:.2%}")