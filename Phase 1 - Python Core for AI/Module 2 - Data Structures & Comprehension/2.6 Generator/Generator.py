from typing import Generator

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Generator[str, None, None]:
    """Yield overlapping text chunks for embedding/RAG pipelines."""
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield text[start:end]
        start += chunk_size - overlap


# Load a long document (simulated)
document = "Python is a versatile language. " * 30

# Process chunks without loading all into memory at once
chunk_count = 0

for chunk in chunk_text(document, chunk_size=100, overlap=20):
    chunk_count += 1
    # In real code: embed the chunk and store in vector DB

print(f"Processed {chunk_count} chunks")


# Generator expression - like a list comprehension but lazy
sizes = (len(chunk) for chunk in chunk_text(document, 100, 20))

print(f"Max chunk size: {max(sizes)}")