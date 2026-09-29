from dataclasses import dataclass
import re


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int


def chunk_by_sentences(
    text: str,
    doc_id: str,
    max_chars: int = 1000,
    overlap_chars: int = 100,
) -> list[Chunk]:
    """
    Split text into chunks that respect sentence boundaries.
    Adds overlap so context is not lost at chunk edges.
    """

    # Split on sentence endings
    sentences = re.split(
        r'(?<=[.!?])\s+',
        text.strip()
    )

    chunks: list[Chunk] = []
    current = ""
    current_start = 0
    char_offset = 0
    chunk_idx = 0

    for sentence in sentences:
        candidate = (
            (current + " " + sentence).strip()
            if current
            else sentence
        )

        if len(candidate) > max_chars and current:

            # Save current chunk
            end = char_offset + len(current)

            chunks.append(
                Chunk(
                    doc_id,
                    chunk_idx,
                    current.strip(),
                    char_offset,
                    end
                )
            )

            chunk_idx += 1

            # Start new chunk with overlap from end of previous
            overlap_start = max(
                0,
                len(current) - overlap_chars
            )

            overlap_text = current[overlap_start:]

            current = (
                overlap_text + " " + sentence
            ).strip()

            char_offset = (
                end - len(overlap_text)
            )

        else:
            current = candidate

    # Save final chunk
    if current.strip():

        end = char_offset + len(current)

        chunks.append(
            Chunk(
                doc_id,
                chunk_idx,
                current.strip(),
                char_offset,
                end
            )
        )

    return chunks


# Test with a sample document
document = """
Large language models (LLMs) are neural networks trained on vast amounts of text data.
They learn to predict the next token in a sequence, which gives them broad language understanding.
Models like GPT-4 and Claude are examples of LLMs used in production today.
Retrieval-Augmented Generation, or RAG, extends LLMs by connecting them to external knowledge bases.
Instead of relying solely on knowledge encoded during training, a RAG system retrieves relevant documents at inference time.
This allows the model to answer questions about recent events or private data it was never trained on.
The retrieval step in RAG typically uses embedding-based semantic search.
A query is embedded into a vector, and the nearest document vectors are retrieved from a database.
These documents are then injected into the LLM's context window alongside the query.
"""


chunks = chunk_by_sentences(
    document.strip(),
    doc_id="intro_to_llms",
    max_chars=300,
    overlap_chars=50
)

for c in chunks:
    print(
        f"Chunk {c.chunk_index} "
        f"({c.char_start}-{c.char_end}): "
        f"{c.text[:80]}..."
    )
    print()