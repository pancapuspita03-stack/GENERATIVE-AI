# ==========================================
# 1. RateLimiter
# ==========================================
import time
from collections import deque


class RateLimiter:
    """Limits the number of calls per minute."""

    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.calls = deque()

    def check_and_wait(self):
        """Wait if the maximum number of calls is reached."""
        now = time.time()

        while self.calls and now - self.calls[0] >= 60:
            self.calls.popleft()

        if len(self.calls) >= self.max_calls:
            wait_time = 60 - (now - self.calls[0])
            print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)

            now = time.time()

            while self.calls and now - self.calls[0] >= 60:
                self.calls.popleft()

        self.calls.append(time.time())


# ==========================================
# 2. PromptTemplate
# ==========================================
from dataclasses import dataclass
import string


@dataclass
class PromptTemplate:
    """Template for generating prompts with placeholders."""

    template: str

    def render(self, **kwargs) -> str:
        """Fill placeholders using str.format_map()."""
        formatter = string.Formatter()

        placeholders = {
            field_name
            for _, field_name, _, _ in formatter.parse(self.template)
            if field_name is not None
        }

        missing = placeholders - kwargs.keys()

        if missing:
            raise ValueError(
                f"Missing placeholders: {', '.join(sorted(missing))}"
            )

        return self.template.format_map(kwargs)


# ==========================================
# 3. Retry Decorator
# ==========================================
import functools


def retry(max_attempts=3, delay=0.1):
    """Retry a function when an exception occurs."""

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as error:
                    last_error = error
                    print(f"Attempt {attempt} failed: {error}")

                    if attempt < max_attempts:
                        time.sleep(delay)

            raise last_error

        return wrapper

    return decorator


call_count = 0


@retry(max_attempts=3)
def test_retry():
    global call_count
    call_count += 1

    if call_count <= 2:
        raise ValueError("Simulated failure")

    return "Function succeeded!"


# ==========================================
# 4. Modules and Packages
# ==========================================
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from my_ai_project import ConversationHistory, LLMConfig


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":

    # ------------------------------------------
    # 1. Test RateLimiter
    # ------------------------------------------
    print("=== 1. RateLimiter Test ===")

    limiter = RateLimiter(max_calls=3)

    for i in range(5):
        limiter.check_and_wait()
        print(f"Call {i + 1}")

    # ------------------------------------------
    # 2. Test PromptTemplate
    # ------------------------------------------
    print("\n=== 2. PromptTemplate Test ===")

    prompt = PromptTemplate(
        "Explain {topic} for {audience}."
    )

    print(
        prompt.render(
            topic="Generative AI",
            audience="students"
        )
    )

    try:
        prompt.render(topic="Generative AI")
    except ValueError as error:
        print(f"Validation: {error}")

    # ------------------------------------------
    # 3. Test Retry
    # ------------------------------------------
    print("\n=== 3. Retry Decorator Test ===")

    print(test_retry())

    # ------------------------------------------
    # 4. Test Modules and Packages
    # ------------------------------------------
    print("\n=== 4. Modules and Packages Test ===")

    config = LLMConfig(
        model="gpt-4",
        temperature=0.7
    )

    print(f"Model: {config.model}")
    print(f"Temperature: {config.temperature}")

    history = ConversationHistory(
        max_turns=5,
        system_prompt="Be concise."
    )

    history.add("user", "What is an embedding?")
    history.add(
        "assistant",
        "An embedding is a vector representation of data."
    )
    history.add("user", "Give a use case.")

    print(history)
    print(f"Messages: {history.to_api_payload()}")
    print(f"Total messages: {len(history)}")