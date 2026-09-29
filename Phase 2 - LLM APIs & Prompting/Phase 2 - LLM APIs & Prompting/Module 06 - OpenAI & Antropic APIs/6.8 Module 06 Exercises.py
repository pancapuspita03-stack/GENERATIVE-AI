import openai
import os
import asyncio
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Exercise 1
# Retry on Rate Limit with Exponential Backoff
# ============================================================

def retry_on_rate_limit(
    client,
    messages,
    max_retries=5,
    system="",
    max_tokens=512
):
    """
    Retry an API call when RateLimitError occurs using OpenRouter/OpenAI.
    """
    api_messages = []
    if system:
        api_messages.append({"role": "system", "content": system})
    api_messages += messages

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="openrouter/free",
                max_tokens=max_tokens,
                messages=api_messages,
            )
            return response

        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt
            print(f"Rate limit reached. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

    raise RuntimeError("Maximum retries exceeded.")


# Mock client for testing
class MockRateLimitClient:
    def __init__(self, failures_before_success=3):
        self.failures_before_success = failures_before_success
        self.attempts = 0

    class Chat:
        def __init__(self, parent):
            self.parent = parent
            self.completions = self.Completions(parent)

        class Completions:
            def __init__(self, parent):
                self.parent = parent

            def create(self, **kwargs):
                self.parent.attempts += 1
                if self.parent.attempts <= self.parent.failures_before_success:
                    import httpx
                    err_response = httpx.Response(429, request=httpx.Request("POST", "http://test"))
                    raise openai.RateLimitError(message="Mock rate limit", response=err_response, body=None)

                return MockResponse()

    @property
    def chat(self):
        return self.Chat(self)


class MockResponse:
    class Choice:
        class Message:
            content = "Mock response after retry."
        message = Message()
    
    choices = [Choice()]
    
    class Usage:
        prompt_tokens = 10
        completion_tokens = 20

    usage = Usage()


# Test Exercise 1
print("=== MENJALANKAN EXERCISE 1 ===")
mock_client = MockRateLimitClient(failures_before_success=3)
response = retry_on_rate_limit(mock_client, messages=[{"role": "user", "content": "Hello"}])
print("Exercise 1 result:")
print(response.choices[0].message.content)
print("\n" + "="*50 + "\n")


# ============================================================
# Exercise 2
# Token Budget Manager
# ============================================================
class BudgetExceeded(Exception):
    pass

class TokenBudgetManager:
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @property
    def total_tokens(self):
        return self.total_input_tokens + self.total_output_tokens

    def add_usage(self, input_tokens: int, output_tokens: int):
        new_total = self.total_tokens + input_tokens + output_tokens
        if new_total > self.max_tokens:
            raise BudgetExceeded(f"Budget exceeded. Limit: {self.max_tokens}, requested: {new_total}")

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def remaining_tokens(self):
        return self.max_tokens - self.total_tokens


# Test Exercise 2
print("=== MENJALANKAN EXERCISE 2 ===")
budget = TokenBudgetManager(max_tokens=1000)
budget.add_usage(200, 150)
print(f"Exercise 2 - Remaining tokens: {budget.remaining_tokens()}")
print("\n" + "="*50 + "\n")


# ============================================================
# Exercise 3
# Compare Multiple Models Concurrently
# ============================================================

async def call_model(client, model: str, prompt: str):
    start_time = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        return {
            "model": model,
            "response_text": response.choices[0].message.content[:50] + "...", # potong string
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "latency_ms": round(latency_ms, 2)
        }
    except Exception as error:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return {
            "model": model,
            "response_text": f"ERROR: {error}",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": round(latency_ms, 2)
        }


async def compare_models_async(client, prompt: str, models: list[str]) -> pd.DataFrame:
    tasks = [call_model(client, model, prompt) for model in models]
    results = await asyncio.gather(*tasks)
    return pd.DataFrame(results)


def compare_models(prompt: str, models: list[str]) -> pd.DataFrame:
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1"
    )
    return asyncio.run(compare_models_async(client, prompt, models))


# Example Exercise 3
print("=== MENJALANKAN EXERCISE 3 ===")
models = ["openrouter/free"]
results = compare_models("Explain what a vector database is.", models)
print("Exercise 3 result:")
print(results.to_string(index=False))
print("\n" + "="*50 + "\n")


# ============================================================
# Exercise 4
# Stream Response Directly to Console (Modified)
# ============================================================

def stream_response(prompt: str):
    client = OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1"
    )

    response = client.chat.completions.create(
        model="openrouter/free",
        max_tokens=512,
        stream=True,
        messages=[{"role": "user", "content": prompt}]
    )
    for chunk in response:
        if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            print(text, end="", flush=True)
            
    print() # Adds an empty line at the end

# Example Exercise 4
print("=== MENJALANKAN EXERCISE 4 ===")
stream_response("Explain retrieval-augmented generation.")
print("\n" + "="*50 + "\n")