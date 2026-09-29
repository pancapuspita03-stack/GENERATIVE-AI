# Building a conversation history
conversation = []

conversation.append({"role": "user", "content": "Hello"})
conversation.append({"role": "assistant", "content": "Hi! Howcan I help?"})
conversation.append({"role": "user", "content": "ExplainRAG."})

print(len(conversation)) # 3
print(conversation[0]) # first message
print(conversation[-1]) # last message
print(conversation[1:3]) # slice: messages 1 and 2

# Useful list methods
scores = [0.91, 0.76, 0.88, 0.65, 0.95]
scores.sort(reverse=True) # in-place sort
print(scores) # [0.95, 0.91, 0.88,0.76, 0.65]
print(max(scores), min(scores))

# Remove by value vs by index
scores.remove(0.76) # remove first occurrence of this value
popped = scores.pop() # remove and return the last element
print(popped, scores)