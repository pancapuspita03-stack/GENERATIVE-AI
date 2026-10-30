# 10.2 Setup and Settings (Halaman 116)
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os
from dotenv import load_dotenv

load_dotenv()

# Configure global settings - applied to all indexes and query engines
Settings.llm = OpenAI(
    model="gpt-4o",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.1,
    max_tokens=1024,
)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
)
Settings.chunk_size = 512       # tokens per chunk
Settings.chunk_overlap = 64     # token overlap between chunks

if __name__ == "__main__":
    print("LlamaIndex Global Settings configured successfully.")
