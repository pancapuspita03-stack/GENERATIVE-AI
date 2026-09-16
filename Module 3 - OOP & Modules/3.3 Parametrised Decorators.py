import functools
import time


def retry(max_attempts: int = 3, delay: float = 0.1):
    """Parametrised retry decorator."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_error

        return wrapper

    return decorator


@retry(max_attempts=3, delay=0.05)
def flaky_api_call(prompt: str) -> str:
    import random

    if random.random() < 0.6:  # fails 60% of the time
        raise ConnectionError("Simulated network error")
    return f"Response to:{prompt}"
# Perintah untuk menampilkan output ke terminal
print(flaky_api_call("Hello World"))