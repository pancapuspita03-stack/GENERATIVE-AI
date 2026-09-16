# Core scalar types
model_name: str = "claude-sonnet-4-5"
temperature: float = 0.7
max_tokens: int = 1024
is_streaming: bool = True
# Check types at runtime
print(type(model_name)) # <class 'str'>
print(type(temperature)) # <class 'float'>
# None - the absence of a value
response = None
print(response is None) # True