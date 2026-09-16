def chunk_text(text: str, chunk_size: int = 100):
    """Memotong teks menjadi beberapa bagian (chunks)."""
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i : i + chunk_size])