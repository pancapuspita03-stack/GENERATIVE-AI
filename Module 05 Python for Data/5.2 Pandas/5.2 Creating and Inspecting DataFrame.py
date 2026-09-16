import pandas as pd

data = [
    {
        "model": "gpt-4o",
        "provider": "OpenAI",
        "context_k": 128,
        "cost_input": 2.50,
    },
    {
        "model": "claude-sonnet-4-5",
        "provider": "Anthropic",
        "context_k": 200,
        "cost_input": 3.00,
    },
    {
        "model": "gemini-1.5-pro",
        "provider": "Google",
        "context_k": 1000,
        "cost_input": 1.25,
    },
    {
        "model": "llama-3.1-70b",
        "provider": "Meta",
        "context_k": 128,
        "cost_input": 0.00,
    },
]

df = pd.DataFrame(data)

print(df.shape)  # Output: (4, 4)
print(df.dtypes)
print(df.head())
print(df.describe())