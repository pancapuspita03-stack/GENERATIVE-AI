# pip install weaviate-client openai python-dotenv
import weaviate
import weaviate.classes as wvc
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
if not openrouter_api_key:
    raise RuntimeError("OPENROUTER_API_KEY belum disetel di file .env atau environment.")

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

def embed(texts: list[str]) -> list[list[float]]:
    response = openrouter_client.embeddings.create(
        model="nvidia/nemotron-3.5-embed-1b:free",
        input=texts,
    )
    return [e.embedding for e in sorted(response.data, key=lambda x: x.index)]

# --- Connect (local Docker or Weaviate Cloud) ---
# Local: docker run -d -p 8080:8080 -p 50051:50051 \
#   cr.weaviate.io/semitechnologies/weaviate:latest
# client = weaviate.connect_to_local()
# Cloud:
# client = weaviate.connect_to_weaviate_cloud(
#     cluster_url=os.environ["WEAVIATE_URL"],
#     auth_credentials=wvc.auth.Auth.api_key(os.environ["WEAVIATE_API_KEY"]),
# )

# --- Create collection (schema) ---
# client.collections.create(
#     name="AIDocument",
#     properties=[
#         wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
#         wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
#         wvc.config.Property(name="category", data_type=wvc.config.DataType.TEXT),
#     ],
#     vectorizer_config=wvc.config.Configure.Vectorizer.none(),
# )

# --- Insert objects with vectors ---