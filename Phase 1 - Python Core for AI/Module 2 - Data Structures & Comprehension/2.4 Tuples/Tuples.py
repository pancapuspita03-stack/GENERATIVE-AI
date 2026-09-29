# Return multiple values from a function
def parse_model_string(model_id: str) -> tuple[str, str, str]:

    """Parse 'provider/model-name:version' into parts."""
    provider, rest = model_id.split("/")
    if ":" in rest:
        name, version = rest.split(":")
    else:
        name, version = rest, "latest"
    return provider, name, version

provider, name, version = parse_model_string("anthropic/claude-sonnet-4-5:20241022")
print(f"Provider:{provider}, Model:{name}, Version:{version}")

# Named tuples - self-documenting tuples
from typing import NamedTuple
class EmbeddingResult(NamedTuple):
    text: str
    vector: list[float]
    model: str
    
result = EmbeddingResult(
    text="Hello world",
    vector=[0.12, -0.34, 0.89],
    model="text-embedding-3-small",
)
print(result.text, result.model)