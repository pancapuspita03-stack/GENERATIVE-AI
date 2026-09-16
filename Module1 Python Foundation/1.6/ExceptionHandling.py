import time
from typing import Optional

class APIError(Exception):
    """Raised when an AI API returns an error response."""
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code
def call_api_with_retry(
    prompt: str,
    max_retries: int = 3,
    backoff_seconds: float = 2.0,
) -> Optional[str]:
    """Call a mock API with exponential backoff on failure."""
    for attempt in range(1, max_retries + 1):
        try:
            # Simulate API call - replace with real client call
            if attempt < 3:
                raise APIError("Rate limit exceeded", 429)
            return f"Response to:{prompt}"
        except APIError as e:
            if e.status_code == 429 and attempt < max_retries:
                wait = backoff_seconds ** attempt
                print(f"Rate limited. Retrying in{wait:.1f}s...")
                time.sleep(0.01) # shortened for demo