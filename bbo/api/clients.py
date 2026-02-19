"""Unified chat completion interface for multiple LLM providers."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path)

# Model name -> (provider, api_model_id)
MODEL_REGISTRY = {
    "mistral-small": ("mistral", "mistral-small-latest"),
    "mistral-large": ("mistral", "mistral-large-latest"),
    "ministral-8b": ("mistral", "ministral-8b-latest"),
    "ministral-3b": ("mistral", "ministral-3b-latest"),
    "gpt-4o-mini": ("openai", "gpt-4o-mini"),
    "claude-sonnet": ("anthropic", "claude-sonnet-4-20250514"),
}


def _call_mistral(model_id: str, system_prompt: str, user_message: str,
                  temperature: float, max_tokens: int) -> str:
    from mistralai import Mistral
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    resp = client.chat.complete(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def _call_openai(model_id: str, system_prompt: str, user_message: str,
                 temperature: float, max_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})
    resp = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def _call_anthropic(model_id: str, system_prompt: str, user_message: str,
                    temperature: float, max_tokens: int) -> str:
    from anthropic import Anthropic
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    kwargs = dict(
        model=model_id,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=max_tokens,
    )
    if system_prompt:
        kwargs["system"] = system_prompt
    if temperature > 0:
        kwargs["temperature"] = temperature
    resp = client.messages.create(**kwargs)
    return resp.content[0].text


_PROVIDERS = {
    "mistral": _call_mistral,
    "openai": _call_openai,
    "anthropic": _call_anthropic,
}


def chat_completion(model: str, system_prompt: str, user_message: str,
                    temperature: float = 0.0, max_tokens: int = 512,
                    max_retries: int = 5) -> str:
    """Call an LLM API with retry logic.

    Parameters
    ----------
    model : str
        One of: mistral-small, mistral-large, gpt-4o-mini, claude-sonnet
    system_prompt : str
        System prompt text (can be empty string for no system prompt)
    user_message : str
        User message / query text
    temperature : float
        Sampling temperature (0 = deterministic)
    max_tokens : int
        Maximum response tokens
    max_retries : int
        Number of retries with exponential backoff

    Returns
    -------
    str : Response text
    """
    if model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model}. Available: {list(MODEL_REGISTRY.keys())}")

    provider, model_id = MODEL_REGISTRY[model]
    call_fn = _PROVIDERS[provider]

    for attempt in range(max_retries):
        try:
            return call_fn(model_id, system_prompt, user_message, temperature, max_tokens)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt + 1
            print(f"  Retry {attempt + 1}/{max_retries} after error: {e}. Waiting {wait}s...")
            time.sleep(wait)
