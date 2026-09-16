messages = [
    {"role": "user", "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."},
    {"role": "user", "content": "Give an example."},
]

# Extract only user messages
user_messages = [m["content"] for m in messages if m["role"]== "user"]
print(user_messages)

# Word count per message
word_counts = [len(m["content"].split()) for m in messages]
print(word_counts) # [3, 7, 3]

# Flatten a nested list
keywords = [["RAG", "retrieval"], ["LLM", "embedding"], ["vector"]]
flat = [kw for group in keywords for kw in group]
print(flat)