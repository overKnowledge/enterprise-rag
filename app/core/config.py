from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Enterprise RAG Platform"
    environment: str = "development"
    log_level: str = "INFO"
    groq_model: str = "openai/gpt-oss-120b"  # fallback default, override via .env

    openai_api_key: SecretStr | None = None
    mistral_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()