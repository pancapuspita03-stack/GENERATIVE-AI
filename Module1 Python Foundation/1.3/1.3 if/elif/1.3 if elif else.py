def classify_response_length(token_count: int) -> str:
    if token_count < 100:
        return "short"
    elif token_count < 500:
        return "medium"
    elif token_count < 2000:
        return "long"
    else:
        return "very long"
print(classify_response_length(80)) # short
print(classify_response_length(350)) # medium
print(classify_response_length(3000)) # very long