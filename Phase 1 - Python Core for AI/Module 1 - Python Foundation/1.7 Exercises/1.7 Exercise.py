from functools import wraps
import time

# ==========================================
# 1. Token Cost Calculator
# ==========================================


def token_cost(tokens: int, model: str) -> float:
    """Calculates total cost using a dictionary of rates per 1K tokens."""
    costs_per_1k = {
        "gpt-4o": 0.0025,
        "claude-sonnet-4-5": 0.0030,
        "gemini-1.5-pro": 0.00125,
        "llama-3.1-70b": 0.0,
    }

    model_key = model.lower()
    if model_key not in costs_per_1k:
        raise ValueError(
            f"Unknown model: '{model}'. Allowed models: {list(costs_per_1k.keys())}"
        )

    return (tokens / 1000) * costs_per_1k[model_key]


# ==========================================
# 2. Retry Decorator
# ==========================================


def retry(n: int = 3, delay: float = 0.01):
    """Decorator that retries a function up to n times on any exception."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, n + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == n:
                        raise e
                    print(
                        f"[Attempt {attempt} failed: {e}] Retrying in {delay}s..."
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


# Mock function to test retry decorator (fails the first two times)
attempt_counter = 0


@retry(n=3, delay=0.01)
def unreliable_api_call():
    global attempt_counter
    attempt_counter += 1
    if attempt_counter < 3:
        raise ConnectionError("API Service Unavailable")
    return "API call successful!"


# ==========================================
# 3. Temperature Label Mapper
# ==========================================


def temperature_label(t: float) -> str:
    """Maps temperature value float to a descriptive string label."""
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Temperature {t} is outside the range 0.0 - 1.0")

    if t <= 0.3:
        return "precise"
    elif t <= 0.7:
        return "balanced"
    else:
        return "creative"


# ==========================================
# 4. String Parsing (Without Regex)
# ==========================================


def parse_token_string(text: str) -> tuple[int, float]:
    """Extracts token count (int) and cost (float) using only string methods."""
    # Input format: "128000 tokens, 0.005 USD per 1K"
    parts = text.split(",")

    # Part 1: "128000 tokens"
    tokens = int(parts[0].strip().split()[0])

    # Part 2: " 0.005 USD per 1K"
    cost = float(parts[1].strip().split()[0])

    return tokens, cost


# ==========================================
# Uji Coba Semua Latihan (Main Execution)
# ==========================================
if __name__ == "__main__":
    print("=== 1. Test Token Cost ===")
    print("Cost for 5000 tokens (gpt-4o): $", token_cost(5000, "gpt-4o"))
    try:
        token_cost(1000, "unknown-model")
    except ValueError as e:
        print("[Expected Error]:", e)

    print("\n=== 2. Test Retry Decorator ===")
    result = unreliable_api_call()
    print("Result:", result)

    print("\n=== 3. Test Temperature Label ===")
    print("Temp 0.2:", temperature_label(0.2))
    print("Temp 0.5:", temperature_label(0.5))
    print("Temp 0.9:", temperature_label(0.9))
    try:
        temperature_label(1.5)
    except ValueError as e:
        print("[Expected Error]:", e)

    print("\n=== 4. Test String Parsing (No Regex) ===")
    raw_str = "128000 tokens, 0.005 USD per 1K"
    token_cnt, cost_val = parse_token_string(raw_str)
    print(f"Parsed -> Tokens: {token_cnt} ({type(token_cnt).__name__}) | Cost: {cost_val} ({type(cost_val).__name__})")