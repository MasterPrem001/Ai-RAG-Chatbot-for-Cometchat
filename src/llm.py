"""
LLM Client Wrapper for Aster & Row Support Agent.

Handles LLM communication using the Gemini API.
"""

import sys
import time
from openai import OpenAI
from src import config

_gemini_client = None

def get_gemini_client() -> OpenAI:
    global _gemini_client
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the environment.")
    
    if _gemini_client is None:
        _gemini_client = OpenAI(
            api_key=config.GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    return _gemini_client

_last_api_call = 0.0

def generate_completion(messages: list[dict], **kwargs) -> str:
    """
    Generates a completion using Gemini.
    """
    global _last_api_call
    client = get_gemini_client()
    
    max_retries = 3
    base_delay = 5
    
    models_to_try = [config.GEMINI_MODEL]
    if config.GEMINI_BACKUP_MODEL:
        models_to_try.append(config.GEMINI_BACKUP_MODEL)
        
    for attempt in range(max_retries):
        # Enforce global rate limit of 15 requests/min (1 per 4 seconds)
        now = time.time()
        elapsed = now - _last_api_call
        if elapsed < 4.5:
            time.sleep(4.5 - elapsed)
        _last_api_call = time.time()
        
        for model_name in models_to_try:
            try:
                kwargs_copy = kwargs.copy()
                if "model" in kwargs_copy:
                    del kwargs_copy["model"]
                
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    **kwargs_copy
                )
                return response.choices[0].message.content
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "quota" in err_str or "rate limit" in err_str:
                    print(f"\n[WARNING]: Gemini API ({model_name}) rate limited.", file=sys.stderr)
                    continue  # Try backup model
                
                print(f"\n[WARNING]: Gemini API ({model_name}) failed ({e}).", file=sys.stderr)
                continue  # Try backup model
        
        # If we exhausted models in this attempt, wait before next attempt
        if attempt == max_retries - 1:
            raise Exception(f"Gemini API failed after {max_retries} attempts.")
        
        print(f"\n[WARNING]: All models failed on attempt {attempt + 1}. Retrying...", file=sys.stderr)
        time.sleep(base_delay * (attempt + 1))

    raise Exception("Failed to generate completion.")
