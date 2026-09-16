from collections import defaultdict

# Track token usage per model
usage: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})

usage["claude-sonnet-4-5"]["input"] += 350
usage["claude-sonnet-4-5"]["output"] += 210
usage["gpt-4o"]["input"] += 420
usage["gpt-4o"]["output"] += 180

for model, counts in usage.items():
    total = counts["input"] + counts["output"]
    print(f"{model}:{total} total tokens")