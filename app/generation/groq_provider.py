import time

from groq import Groq, RateLimitError

from app.core.config import get_settings


class GroqLLMProvider:
    def __init__(self, model: str | None = None):
        settings = get_settings()
        if settings.groq_api_key is None:
            raise ValueError("GROQ_API_KEY is not set in .env")
        self._client = Groq(api_key=settings.groq_api_key.get_secret_value())
        self._model = model or settings.groq_model

    def generate(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                last_error = e
                wait_seconds = 5 * (attempt + 1)  # simple backoff: 5s, 10s, 15s
                print(
                    f"  Rate limited, waiting {wait_seconds}s "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_seconds)
        raise last_error    