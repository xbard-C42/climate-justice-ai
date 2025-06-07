# earth_justice/types.py
from pydantic import BaseModel
from typing import List, Dict, Any

class LLMError(Exception):
    """Base class for all LLM-related errors."""
    pass

class ProviderUnavailableError(LLMError):
    pass

class LLMTrafficError(LLMError):
    pass

class ContextWindowExhaustionError(LLMError):
    pass

class AIRequest(BaseModel):
    origin_region: str
    purpose: str
    emergency_level: int
    prompt: List[Dict[str, Any]]  # or str for single-string formats
    model: str

class AIResponse(BaseModel):
    content: str
    usage: Dict[str, int]