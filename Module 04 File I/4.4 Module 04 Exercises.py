import asyncio
import csv
from datetime import datetime, timezone
import json
import os
import pathlib
import threading
import time
from dotenv import load_dotenv
import httpx

# ==========================================
# 1. JSON Serialization/Deserialization
# ==========================================


def save_conversation(history: list[dict], path: str) -> None:
    """Serialise conversation history to a JSON file."""
    file_path = pathlib.Path(path)
    file_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_conversation(path: str) -> list[dict]:
    """Deserialise conversation history from a JSON file."""
    file_path = pathlib.Path(path)
    if not file_path.exists():
        return []
    return json.loads(file_path.read_text(encoding="utf-8"))


# ==========================================
# 2. Concurrent Async Endpoint Comparison
# ==========================================


async def fetch_endpoint(
    client: httpx.AsyncClient, url: str
) -> tuple[str, int, float]:
    """Fetch single URL and measure response time."""
    start_time = time.perf_counter()
    try:
        response = await client.get(url, timeout=10.0)
        status_code = response.status_code
    except httpx.HTTPError:
        status_code = 0  # Signal failure/timeout

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return (url, status_code, round(elapsed_ms, 2))


async def compare_endpoints(urls: list[str]) -> list[tuple[str, int, float]]:
    """Fetches all URLs concurrently and returns (url, status_code, response_time_ms)."""
    async with httpx.AsyncClient() as client:
        tasks = [fetch_endpoint(client, url) for url in urls]
        return await asyncio.gather(*tasks)


# ==========================================
# 3. Config Loader with Env Overrides
# ==========================================


def load_config_with_env_overrides(config_path: str) -> dict:
    """Reads JSON config file and merges with environment variable overrides."""
    load_dotenv()
    path = pathlib.Path(config_path)

    config = {}
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))

    # Env vars take precedence
    for key in list(config.keys()):
        env_val = os.getenv(key.upper())
        if env_val is not None:
            # Type casting basic values if needed
            if env_val.isdigit():
                config[key] = int(env_val)
            elif env_val.replace(".", "", 1).isdigit():
                config[key] = float(env_val)
            else:
                config[key] = env_val

    return config


# ==========================================
# 4. Thread-Safe CSV Log Writer
# ==========================================


class LLMLogger:
    """Thread-safe CSV logger for LLM calls."""

    def __init__(self, filename: str = "llm_calls.csv"):
        self.file_path = pathlib.Path(filename)
        self.lock = threading.Lock()
        self.fieldnames = [
            "timestamp",
            "model",
            "input_tokens",
            "output_tokens",
            "latency_ms",
        ]

        # Create file and write headers if it doesn't exist
        if not self.file_path.exists():
            with self.lock, self.file_path.open(
                "w", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
    ) -> None:
        """Appends a row thread-safely."""
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 2),
        }
        with self.lock, self.file_path.open(
            "a", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


# ==========================================
# Uji Coba Semua Latihan (Main Execution)
# ==========================================

if __name__ == "__main__":
    print("=== 1. Test Save & Load Conversation ===")
    sample_history = [
        {"role": "user", "content": "Halo AI!"},
        {"role": "assistant", "content": "Halo! Ada yang bisa saya bantu?"},
    ]
    save_conversation(sample_history, "chat.json")
    loaded_history = load_conversation("chat.json")
    print("Loaded Conversation:", loaded_history)

    print("\n=== 2. Test Compare Endpoints ===")
    test_urls = [
        "https://httpbin.org/delay/1",
        "https://jsonplaceholder.typicode.com/posts/1",
    ]
    results = asyncio.run(compare_endpoints(test_urls))
    for res in results:
        print(f"URL: {res[0]} | Status: {res[1]} | Latency: {res[2]}ms")

    print("\n=== 3. Test Config Loader ===")
    sample_config = {"model": "gpt-4o", "temperature": 0.7}
    pathlib.Path("config.json").write_text(json.dumps(sample_config))
    os.environ["MODEL"] = "claude-sonnet-4-5"  # Mock environment override
    merged_config = load_config_with_env_overrides("config.json")
    print("Merged Config:", merged_config)

    print("\n=== 4. Test Thread-Safe CSV Logger ===")
    logger = LLMLogger("llm_calls.csv")
    logger.log("claude-sonnet-4-5", 150, 420, 312.5)
    print("Log berhasil ditulis ke llm_calls.csv")