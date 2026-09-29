# Building an LLM API payload
payload = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 1024,
    "temperature": 0.7,
    "messages": [
        {"role": "user", "content": "What is attention in transformers?"}
    ],
}
# Access
print(payload["model"]) # claude-sonnet-4-5
print(payload.get("top_p", 1.0)) # default 1.0 if key missing

# Update
payload["temperature"] = 0.3
payload.update({"stream": True, "top_k": 40})
# Iteration
for key, value in payload.items():
    if not isinstance(value, list):
        print(f"{key}:{value}")
        
# Membership check
print("stream" in payload) # True
print("top_p" in payload) # False