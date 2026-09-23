import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from groq import Groq  # noqa: E402

from app.core.config import get_settings  # noqa: E402

if __name__ == "__main__":
    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key.get_secret_value())
    models = client.models.list()
    for m in models.data:
        print(m.id)