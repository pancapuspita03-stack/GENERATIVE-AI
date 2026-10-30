# 9.6 Document Loaders and Text Splitters (Halaman 109-110)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Simulate loading documents
raw_docs = [
    Document(
        page_content="""LangChain provides a standard interface for interacting with
large language models and related tools. It simplifies the development of
context-aware reasoning applications by combining prompts, models, memory,
and retrieval in composable pipelines. LCEL makes these pipelines
declarative and easy to debug.""",
        metadata={"source": "langchain_docs.txt", "page": 1}
    ),
    Document(
        page_content="""Retrieval-Augmented Generation (RAG) is a pattern that combines
a retriever with a generator. The retriever finds relevant documents from a
knowledge base. Those documents are injected into the generator's context window.
This allows models to answer questions about information they were never trained on.""",
        metadata={"source": "rag_guide.txt", "page": 1}
    ),
]

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

if __name__ == "__main__":
    chunks = splitter.split_documents(raw_docs)
    print(f"Original docs: {len(raw_docs)}, Chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f" Chunk {i}: {len(chunk.page_content)} chars | source: {chunk.metadata['source']}")
