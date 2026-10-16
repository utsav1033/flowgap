"""
Shared Gemini setup for labeling and node drafting.
Configured lazily so the pipeline still runs without an API key — labels fall
back to keyword extraction and node specs to a template.
"""

import os

from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.0-flash"
_PLACEHOLDER_KEY = "your_gemini_api_key_here"

_enabled = True
_configured = False


def disable():
    """Force the no-LLM path for this process (e.g. `flowgap analyze --no-llm`)."""
    global _enabled
    _enabled = False


def available() -> bool:
    global _configured
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not _enabled or not key or key == _PLACEHOLDER_KEY:
        return False
    if not _configured:
        import google.generativeai as genai
        genai.configure(api_key=key)
        _configured = True
    return True


def generate(prompt: str, temperature: float, max_output_tokens: int) -> str:
    """Single Gemini call. Raises on API errors so callers can retry or fall back."""
    import google.generativeai as genai
    model = genai.GenerativeModel(model_name=MODEL)
    resp = model.generate_content(
        prompt,
        generation_config={"temperature": temperature, "max_output_tokens": max_output_tokens},
    )
    return resp.text.strip()
