# 11.3 Pinecone
from pinecone import Pinecone, ServerlessSpec
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise RuntimeError("OPENROUTER_API_KEY belum disetel di file .env atau environment.")

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)
embedding_model = os.environ.get(
    "OPENROUTER_EMBEDDING_MODEL",
    "nvidia/nemotron-3.5-embed-1b:free",
)

def embed(texts: list[str]) -> list[list[float]]:
    response = openrouter_client.embeddings.create(
        model=embedding_model,
        input=texts,
    )
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

if __name__ == "__main__":
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if pinecone_api_key:
        pc = Pinecone(api_key=pinecone_api_key)
        INDEX_NAME = "ai-knowledge-base"
        documents = [
            {"id": "d1", "text": "RAG connects LLMs to external knowledge bases."},
            {"id": "d2", "text": "Fine-tuning updates model weights on task-specific data."},
            {"id": "d3", "text": "ChromaDB is an open-source local vector database."},
        ]
        document_embeddings = embed([doc["text"] for doc in documents])
        embedding_dimension = len(document_embeddings[0])

        # Create index if it doesn't exist
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        index = pc.Index(INDEX_NAME)

        vectors = [
            {
                "id": doc["id"],
                "values": embedding,
                "metadata": {"text": doc["text"], "source": "guide"},
            }
            for doc, embedding in zip(documents, document_embeddings)
        ]
        index.upsert(vectors=vectors)
        print(index.describe_index_stats())

        # Query
        q_vector = embed(["how to add external knowledge to an LLM?"])[0]
        results = index.query(
            vector=q_vector,
            top_k=3,
            include_metadata=True,
        )
        for match in results["matches"]:
            print(f"[{match['score']:.4f}] {match['metadata']['text']}")
    else:
        print("PINECONE_API_KEY not configured. Set it in .env to run live Pinecone calls.")