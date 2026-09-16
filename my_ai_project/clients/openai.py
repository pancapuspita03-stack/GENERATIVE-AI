class OpenAIClient:
    """Simulates an OpenAI API client for testing."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str) -> str:
        return f"[Mock OpenAI - {self.model}] Response to: {prompt}"