from typing import Protocol


class LLMProvider(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's text completion for the given prompts."""
        ...