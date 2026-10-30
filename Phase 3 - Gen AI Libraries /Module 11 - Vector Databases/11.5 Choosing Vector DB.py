# 11.5 Choosing a Vector Database (Halaman 133)
"""
Requirement                         | Best choice
------------------------------------+--------------------
Local development, simple setup     | ChromaDB
Production, managed, no ops         | Pinecone
Hybrid BM25 + vector search         | Weaviate
Already using PostgreSQL            | pgvector
Maximum scale, self-hosted          | Qdrant or Weaviate

Decision rule: Start with ChromaDB in development. When you go to production,
choose Pinecone if you want zero infrastructure work, or Weaviate/Qdrant if you
need hybrid search or want to self-host.
"""

def print_decision_rules():
    print("Dev: ChromaDB | Cloud Managed: Pinecone | Hybrid Search: Weaviate | Postgres: pgvector")

if __name__ == "__main__":
    print_decision_rules()
