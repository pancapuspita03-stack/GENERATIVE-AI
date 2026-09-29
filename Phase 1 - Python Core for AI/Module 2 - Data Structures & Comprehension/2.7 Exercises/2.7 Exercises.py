# ==========================================
# 1. API Response Filtering & Sorting
# ==========================================
responses = [
    {"tokens": 450, "latency_ms": 620},
    {"tokens": 120, "latency_ms": 340},
    {"tokens": 300, "latency_ms": 480},
    {"tokens": 80, "latency_ms": 210},
    {"tokens": 500, "latency_ms": 490},
]

# Filtering latency < 500ms dan diurutkan berdasarkan tokens ascending
fast_responses = sorted(
    [r for r in responses if r["latency_ms"] < 500],
    key=lambda x: x["tokens"],
)

# ==========================================
# 2. Conversation Stats Function
# ==========================================


def conversation_stats(messages: list[dict]) -> dict:
    """Returns statistics about total messages, user/assistant turns, and average words."""
    if not messages:
        return {
            "total_messages": 0,
            "user_turns": 0,
            "assistant_turns": 0,
            "avg_words_per_message": 0.0,
        }

    total_messages = len(messages)
    user_turns = sum(1 for m in messages if m.get("role") == "user")
    assistant_turns = sum(1 for m in messages if m.get("role") == "assistant")

    total_words = sum(len(m.get("content", "").split()) for m in messages)
    avg_words = total_words / total_messages

    return {
        "total_messages": total_messages,
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "avg_words_per_message": round(avg_words, 2),
    }


# ==========================================
# 3. Batch Generator Function
# ==========================================


def batch_items(items, batch_size: int):
    """Yields lists of batch_size items from any iterable, handling partial batches."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ==========================================
# 4. Set Intersection Operations
# ==========================================
fast_models = ["gpt-4o-mini", "claude-3-haiku", "gemini-1.5-flash", "llama-3-8b"]
cheap_models = [
    "claude-3-haiku",
    "gpt-3.5-turbo",
    "gemini-1.5-flash",
    "mistral-7b",
]

# Menggunakan set intersection (&)
fast_and_cheap = list(set(fast_models) & set(cheap_models))


# ==========================================
# Uji Coba Semua Latihan (Main Execution)
# ==========================================
if __name__ == "__main__":
    print("=== 1. Filtered API Responses (<500ms, sorted by tokens) ===")
    for r in fast_responses:
        print(r)

    print("\n=== 2. Conversation Stats ===")
    sample_chat = [
        {"role": "user", "content": "What is Python generator?"},
        {
            "role": "assistant",
            "content": "A generator is a function that yields values one by one.",
        },
        {"role": "user", "content": "Give me an example code."},
    ]
    stats = conversation_stats(sample_chat)
    print(stats)

    print("\n=== 3. Batch Items Generator ===")
    data_list = [10, 20, 30, 40, 50, 60, 70]
    for b in batch_items(data_list, batch_size=3):
        print("Batch:", b)

    print("\n=== 4. Fast & Cheap Models ===")
    print("Models:", fast_and_cheap)