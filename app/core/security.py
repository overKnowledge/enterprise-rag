from typing import Annotated

from fastapi import Header, HTTPException

from app.core.config import get_settings


def verify_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    settings = get_settings()
    if settings.api_key is None:
        return  # auth disabled if no key configured (e.g. local dev)
    if x_api_key is None or x_api_key != settings.api_key.get_secret_value():
        raise HTTPException(status_code=401, detail="Invalid or missing API key")