def build_prompt(system: str, user: str, temperature: float =0.7) -> str:
    """Assemble a prompt string for an LLM API call.

    Args:
        system: The system instruction.
        user: The user message.
        temperature: Sampling temperature (0.0 – 2.0).

    Returns:
        Formatted prompt string.
        """
    return f"[System]\n{system}\n\n[User]\n{user}"
result = build_prompt(
system="You are a concise AI assistant.",
user="What is backpropagation?",
)
print(result)