import functools
import time


def log_call(func):
    """Decorator: log function name and return value."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f">>> Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"<<< {func.__name__} returned: {result!r}")
        return result

    return wrapper


def cache_result(func):
    """Simple in-memory cache (no expiry)."""
    _cache: dict = {}

    @functools.wraps(func)
    def wrapper(*args):
        if args not in _cache:
            _cache[args] = func(*args)
        return _cache[args]

    return wrapper


@log_call
@cache_result
def get_embedding(text: str) -> list[float]:
    """Simulate an embedding API call (cached)."""
    time.sleep(0.01)  # simulate latency
    return [hash(text) % 100 / 100.0, 0.42, 0.87]


# First call: logs + computes
e1 = get_embedding("What is RAG?")

# Second call: logs but returns from cache instantly
e2 = get_embedding("What is RAG?")

print(e1 == e2)  # True