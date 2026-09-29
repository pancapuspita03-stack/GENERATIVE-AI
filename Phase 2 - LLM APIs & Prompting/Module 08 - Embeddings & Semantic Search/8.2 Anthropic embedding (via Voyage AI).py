# pip install voyageai
import voyageai
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    # Initialize Voyage AI client
    vo = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))

    print("Requesting embeddings from Voyage AI...")

    # Fetch embeddings
    result = vo.embed(
        ["What is RAG?", "Explain vector databases."],
        model="voyage-3",       # current recommended model
        input_type="document",  # "document" for corpus, "query" for search queries
    )

    # Convert to Numpy array
    embeddings = np.array(result.embeddings, dtype=np.float32)

    print(f"Shape: {embeddings.shape}")  # (2, 1024)
    print(f"Token usage: {result.total_tokens}")

except Exception as e:
    print("\n[ERROR] Failed to fetch Embeddings.")
    print("Please ensure your VOYAGE_API_KEY is valid and has sufficient credits.")
    print(f"Original Error: {str(e)}")
