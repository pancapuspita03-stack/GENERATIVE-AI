# 14.3 Tools for Agents (Halaman 155-156)
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import json

class TokenCostCalculatorInput(BaseModel):
    model: str = Field(description="LLM model name, e.g. 'gpt-4o'")
    input_tokens: int = Field(description="Number of input tokens")
    output_tokens: int = Field(description="Number of output tokens")

class TokenCostCalculator(BaseTool):
    name: str = "Token Cost Calculator"
    description: str = "Calculate the USD cost of an LLM API call given model name and token counts."
    args_schema: Type[BaseModel] = TokenCostCalculatorInput

    def _run(self, model: str, input_tokens: int, output_tokens: int) -> str:
        pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
            "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
        }
        if model not in pricing:
            return f"Unknown model: {model}"
        p = pricing[model]
        cost = (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000
        return json.dumps({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 6),
        })

class ModelContextLookupInput(BaseModel):
    model: str = Field(description="LLM model name")

class ModelContextLookup(BaseTool):
    name: str = "Model Context Window Lookup"
    description: str = "Look up the context window size (in tokens) of an LLM."
    args_schema: Type[BaseModel] = ModelContextLookupInput

    def _run(self, model: str) -> str:
        windows = {
            "gpt-4o": 128_000,
            "gpt-4o-mini": 128_000,
            "claude-sonnet-4-5": 200_000,
            "claude-opus-4-5": 200_000,
            "gemini-1.5-pro": 1_000_000,
        }
        tokens = windows.get(model, "Unknown")
        return json.dumps({"model": model, "context_window_tokens": tokens})

cost_tool = TokenCostCalculator()
context_tool = ModelContextLookup()

if __name__ == "__main__":
    print("Test cost tool:", cost_tool._run("gpt-4o", 1000, 500))
    print("Test context tool:", context_tool._run("claude-sonnet-4-5"))
