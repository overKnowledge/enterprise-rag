import pytest
from fastapi import HTTPException
from pydantic import SecretStr

from app.core import security


class _FakeSettings:
    def __init__(self, api_key: str | None):
        self.api_key = SecretStr(api_key) if api_key is not None else None


def test_correct_key_is_accepted(monkeypatch):
    monkeypatch.setattr(security, "get_settings", lambda: _FakeSettings("secret123"))
    security.verify_api_key(x_api_key="secret123")  # should not raise


def test_wrong_key_is_rejected(monkeypatch):
    monkeypatch.setattr(security, "get_settings", lambda: _FakeSettings("secret123"))
    with pytest.raises(HTTPException) as exc_info:
        security.verify_api_key(x_api_key="wrong-key")
    assert exc_info.value.status_code == 401


def test_missing_key_is_rejected(monkeypatch):
    monkeypatch.setattr(security, "get_settings", lambda: _FakeSettings("secret123"))
    with pytest.raises(HTTPException) as exc_info:
        security.verify_api_key(x_api_key=None)
    assert exc_info.value.status_code == 401


def test_no_configured_key_disables_auth(monkeypatch):
    monkeypatch.setattr(security, "get_settings", lambda: _FakeSettings(None))
    security.verify_api_key(x_api_key=None)  # should not raise