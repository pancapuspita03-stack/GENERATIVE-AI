# 10.3 Documents and Nodes (Halaman 116-118)
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

# Create documents from text
documents = [
    Document(
        text="""LlamaIndex is a data framework for LLM applications.
It provides tools for ingesting, structuring, and querying data
using large language models. The core abstraction is the Index,
which organises documents into a structure optimised for retrieval.""",
        metadata={"source": "llamaindex_overview", "category": "framework"},
    ),
    Document(
        text="""Vector store indexes store document embeddings in a vector database.
They support fast semantic search using approximate nearest-neighbour algorithms.
LlamaIndex integrates with ChromaDB, Pinecone, Weaviate, and many other backends.""",
        metadata={"source": "vector_store_guide", "category": "retrieval"},
    ),
    Document(
        text="""The query engine is the highest-level abstraction in LlamaIndex.
It takes a natural language question, retrieves relevant nodes,
and uses an LLM to synthesise a coherent answer from the retrieved context.""",
        metadata={"source": "query_engine_docs", "category": "querying"},
    ),
]

# Parse documents into nodes (chunks)
parser = SentenceSplitter(chunk_size=128, chunk_overlap=20)
nodes = parser.get_nodes_from_documents(documents)

if __name__ == "__main__":
    print(f"Documents: {len(documents)}, Nodes: {len(nodes)}")
    for n in nodes:
        print(f" Node {n.node_id[:8]}... | {len(n.text)} chars | source: {n.metadata.get('source')}")
