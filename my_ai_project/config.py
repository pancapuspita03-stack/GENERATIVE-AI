from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """Configuration for a single LLM API call."""

    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    system_prompt: Optional[str] = None

    def __post_init__(self):
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be 0–2, got {self.temperature}"
            )

        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")

    @property
    def as_dict(self) -> dict:
        d = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        if self.stop_sequences:
            d["stop_sequences"] = self.stop_sequences

        if self.system_prompt:
            d["system"] = self.system_prompt

        return d