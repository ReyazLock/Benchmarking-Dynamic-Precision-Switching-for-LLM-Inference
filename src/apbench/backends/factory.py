from __future__ import annotations

from ..config import ModelConfig
from .anyprecision import AnyPrecisionBackend
from .base import Backend
from .huggingface import HuggingFaceBackend
from .ollama import OllamaBackend


def create_backend(config: ModelConfig) -> Backend:
    if config.backend == "ollama":
        return OllamaBackend(config)
    if config.backend == "anyprecision":
        return AnyPrecisionBackend(config)
    if config.backend == "huggingface":
        return HuggingFaceBackend(config)
    raise ValueError(f"Unsupported backend: {config.backend}")
