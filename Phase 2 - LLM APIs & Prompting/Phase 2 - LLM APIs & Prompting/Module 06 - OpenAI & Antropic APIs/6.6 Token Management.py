import os
from dotenv import load_dotenv

load_dotenv()

# Pada OpenAI / OpenRouter, tidak ada fungsi API bawaan untuk menghitung token secara gratis.
# Perhitungan token biasanya dilakukan secara lokal menggunakan library 'tiktoken',
# atau menggunakan perkiraan matematis kasar (1 kata = ~1.3 token)

prompt_text = "Explain the transformer architecture."
estimated_input = int(len(prompt_text.split()) * 1.3)

print(f"Estimated input tokens: {estimated_input}")

# Cost estimator
PRICING = {
    "openrouter/free": {"input": 0.00, "output": 0.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return estimated cost in USD."""
    if model not in PRICING:
        raise ValueError(f"Unknown model: {model}")

    p = PRICING[model]

    return (
        input_tokens * p["input"] +
        output_tokens * p["output"]
    ) / 1_000_000

cost = estimate_cost(
    "claude-sonnet-4-5",
    input_tokens=500,
    output_tokens=300
)

print(f"Estimated cost for Claude: ${cost:.6f}")

# Context window limits (always check before sending long documents)
CONTEXT_LIMITS = {
    "openrouter/free": 8_000,
    "claude-sonnet-4-5": 200_000,
    "gpt-4o": 128_000,
}

def fits_in_context(
    model: str,
    token_count: int,
    reserve_for_output: int = 2048
) -> bool:
    limit = CONTEXT_LIMITS.get(model, 128_000)
    return token_count + reserve_for_output <= limit