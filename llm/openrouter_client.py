import requests
import json
from typing import List, Dict, Any
from config import OPENROUTER_API_KEY

BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-oss-20b:free"


def call_openrouter(messages: List[Dict[str, Any]], model: str = DEFAULT_MODEL, reasoning: bool = True, timeout: int = 30) -> Dict[str, Any]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set in environment or .env")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
    }

    if reasoning:
        payload["reasoning"] = {"enabled": True}

    resp = requests.post(BASE_URL, headers=headers, data=json.dumps(payload), timeout=timeout)
    resp.raise_for_status()
    return resp.json()
