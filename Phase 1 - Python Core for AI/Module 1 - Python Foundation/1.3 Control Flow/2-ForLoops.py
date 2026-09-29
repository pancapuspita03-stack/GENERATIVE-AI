models = ["gpt-4o", "claude-sonnet-4-5", "gemini-1.5-pro"]
# Basic iteration
for model in models:
    print(f"Checking:{model}")
# With index - use enumerate, not range(len(...))
for i, model in enumerate(models):
    print(f"{i + 1}.{model}")
# Iterate over key-value pairs in a dict
token_limits = {"gpt-4o": 128000, "claude-sonnet-4-5": 200000}
for model, limit in token_limits.items():
    print(f"{model}:{limit:,} tokens")