from groq import Groq

from app.core.config import get_settings


class GroqLLMProvider:
    def __init__(self, model: str | None = None):
        settings = get_settings()
        if settings.groq_api_key is None:
            raise ValueError("GROQ_API_KEY is not set in .env")
        self._client = Groq(api_key=settings.groq_api_key.get_secret_value())
        self._model = model or settings.groq_model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content