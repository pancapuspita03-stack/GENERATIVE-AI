# 12.2 The Pipeline API (Halaman 134-137)
from transformers import pipeline

if __name__ == "__main__":
    # 1. Text classification (sentiment analysis)
    classifier = pipeline(
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1, # -1 = CPU; 0 = first GPU
    )
    texts = [
        "This LLM API is incredibly fast and easy to use.",
        "The documentation is confusing and the latency is terrible.",
        "The model works fine for most tasks.",
    ]
    results = classifier(texts)
    print("=== Sentiment Classification ===")
    for text, result in zip(texts, results):
        print(f"[{result['label']} | {result['score']:.3f}] {text[:60]}")

    # 2. Named entity recognition
    ner = pipeline(
        task="ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
        device=-1,
    )
    text = "Anthropic was founded in San Francisco by Dario Amodei and Daniela Amodei."
    entities = ner(text)
    print("\n=== Named Entity Recognition ===")
    for ent in entities:
        print(f"{ent['word']!r:25s} -> {ent['entity_group']} ({ent['score']:.3f})")

    # 3. Text generation (small local model)
    generator = pipeline(
        task="text-generation",
        model="distilgpt2",
        device=-1,
    )
    output = generator(
        "The future of AI agents is",
        max_new_tokens=50,
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True,
        pad_token_id=50256,
    )
    print("\n=== Text Generation ===")
    print(output[0]['generated_text'])
