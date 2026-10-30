# 12.3 Tokenizers (Halaman 137-138)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
text = "Retrieval-Augmented Generation (RAG) combines search with LLMs."

tokens = tokenizer.encode(text)
print(f"Token count: {len(tokens)}")
print(f"Token IDs: {tokens}")

decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")

token_strings = tokenizer.convert_ids_to_tokens(tokens)
print(f"Tokens: {token_strings}")

# Batch encoding with padding and truncation
batch = ["What is RAG?", "Explain vector databases in detail."]
encoded = tokenizer(
    batch,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
print(f"\nBatch input_ids shape: {encoded['input_ids'].shape}")
print(f"Attention mask shape: {encoded['attention_mask'].shape}")

def count_tokens(text: str, model: str = "gpt2") -> int:
    tok = AutoTokenizer.from_pretrained(model)
    return len(tok.encode(text))

prompt = "Summarise the following document: " + "word " * 50
print(f"Prompt token count: {count_tokens(prompt)}")
