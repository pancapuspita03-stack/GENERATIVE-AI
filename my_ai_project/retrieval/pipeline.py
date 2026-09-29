from my_ai_project.clients.anthropic import AnthropicClient
from my_ai_project.config import LLMConfig
from my_ai_project.retrieval.chunker import chunk_text
import my_ai_project.retrieval.embedder as embedder


def run_pipeline(text: str):
    # 1. Split text into chunks
    chunks = list(chunk_text(text))

    # 2. Generate embeddings for chunks
    vectors = embedder.embed_texts(chunks)

    # 3. Setup LLM configuration & client
    config = LLMConfig(
        model="claude-sonnet-4-5",
        temperature=0.3,
        system_prompt="Be concise.",
    )
    client = AnthropicClient(api_key="mock-key", model=config.model)

    # 4. Generate LLM response
    response = client.complete(text)

    return {
        "chunks": chunks,
        "vectors": vectors,
        "response": response,
        "config": config.as_dict,
    }


if __name__ == "__main__":
    result = run_pipeline("What is an embedding?")
    print(result)