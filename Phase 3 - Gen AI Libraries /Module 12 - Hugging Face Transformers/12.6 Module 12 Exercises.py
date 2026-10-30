# 12.6 Module 12 Exercises (Halaman 143)
from transformers import AutoTokenizer

# Exercise 2: TokenBudgetChecker
class TokenBudgetChecker:
    def __init__(self, model_name: str = "gpt2", max_tokens: int = 100):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens

    def filter_texts(self, texts: list[str]) -> list[tuple[str, int]]:
        passed = []
        for t in texts:
            count = len(self.tokenizer.encode(t))
            if count <= self.max_tokens:
                passed.append((t, count))
        return passed

if __name__ == "__main__":
    checker = TokenBudgetChecker(max_tokens=20)
    samples = [
        "Short sentence.",
        "This sentence is intentionally longer than the twenty-token budget. " * 3,
    ]
    print(checker.filter_texts(samples))
