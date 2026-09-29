from openai import OpenAI, RateLimitError
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def embed(
    texts: list[str],
    model: str = "openai/text-embedding-3-small"
) -> np.ndarray:
    """Embed a list of texts using OpenAI via OpenRouter."""
    
    print(f"Requesting embeddings from OpenAI using model: {model}...")

    # API accepts up to 2048 texts per call
    response = client.embeddings.create(
        input=texts,
        model=model
    )

    # Sort by index to guarantee order matches input
    vectors = sorted(
        response.data,
        key=lambda e: e.index
    )

    return np.array(
        [v.embedding for v in vectors],
        dtype=np.float32
    )


texts = [
    "Retrieval-Augmented Generation combines search with LLMs.",
    "RAG retrieves documents then generates an answer from them.",
    "The Eiffel Tower is in Paris.",
    "Python is a popular programming language.",
    "Fine-tuning trains a model on new data.",
]


try:
    embeddings = embed(texts)
    
    print(f"\nSuccess! Shape: {embeddings.shape}")
    print(f"Norm of first vector: {np.linalg.norm(embeddings[0]):.4f}")  # ≈ 1.0

except RateLimitError as e:
    print("\n[ERROR] OpenAI API Rate Limit Exceeded or Insufficient Quota.")
    print("This means your OpenAI account does not have a paid balance.")
    print("Please add credits to your OpenAI account to use their embedding models.")
    print(f"Original Error: {str(e)}")
except Exception as e:
    print(f"\n[ERROR] An unexpected error occurred: {str(e)}")