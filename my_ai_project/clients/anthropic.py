class AnthropicClient:
    """Simulates an Anthropic API client for testing."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def complete(self, prompt: str) -> str:
        return f"[Mock Anthropic - {self.model}] Response to: {prompt}"