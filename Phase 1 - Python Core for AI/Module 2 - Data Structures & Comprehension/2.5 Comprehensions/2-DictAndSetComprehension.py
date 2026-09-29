models = ["gpt-4o", "claude-sonnet-4-5", "gemini-1.5-pro"]
context_windows = [128_000, 200_000, 1_000_000]

# Dict comprehension
model_context = {m: c for m, c in zip(models, context_windows)}
print(model_context)

# Filter to models with > 150K context
large_context = {m: c for m, c in model_context.items() if c> 150_000}
print(large_context)

# Set comprehension - unique word lengths
sentence = "the quick brown fox jumps over the lazy dog"
unique_lengths = {len(w) for w in sentence.split()}
print(sorted(unique_lengths)) # [2, 3, 4, 5]