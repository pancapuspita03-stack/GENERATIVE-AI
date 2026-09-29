import asyncio


async def call_model_mock(model: str, prompt: str) -> dict:
    """Simulate calling an LLM API (replace with real client call)."""
    await asyncio.sleep(0.05)  # simulate network latency
    return {
        "model": model,
        "response": f"[{model}] Answer to:{prompt[:30]}",
        "tokens": 42,
    }


async def compare_models(prompt: str, models: list[str]) -> list[dict]:
    """Call multiple models in parallel and return all responses."""
    tasks = [call_model_mock(m, prompt) for m in models]
    results = await asyncio.gather(*tasks)
    return list(results)


responses = asyncio.run(
    compare_models(
        "What is RAG?",
        ["claude-sonnet-4-5", "gpt-4o", "gemini-1.5-pro"],
    )
)

for r in responses:
    print(f"{r['model']}:{r['response']}")