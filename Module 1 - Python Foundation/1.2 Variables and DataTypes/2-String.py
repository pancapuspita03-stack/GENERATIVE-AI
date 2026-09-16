user_input = "Explain transformers in simple terms"
system_prompt = "You are a helpful AI tutor."
# f-strings - the preferred way to assemble prompts
full_prompt = f"System:{system_prompt}\nUser:{user_input}"
print(full_prompt)
# Common string methods
print(user_input.upper()) # EXPLAIN TRANSFORMERS...
print(user_input.split()) # ['Explain', 'transformers', ...]
print(user_input.replace("simple", "plain"))
print(len(user_input)) # 35
# Multi-line strings - useful for long system prompts
prompt = """
You are an expert data scientist.
Answer concisely in bullet points.
"""
print(prompt.strip())