from __future__ import annotations

from ..config import ModelConfig
from .anyprecision import AnyPrecisionBackend
from .base import Backend
from .huggingface import HuggingFaceBackend


def create_backend(config: ModelConfig) -> Backend:
    if config.backend == "anyprecision":
        return AnyPrecisionBackend(config)
    if config.backend == "huggingface":
        return HuggingFaceBackend(config)
    if config.backend == "pmpd":
        from .pmpd import PMPDBackend

        return PMPDBackend(config)
    raise ValueError(f"Unsupported backend: {config.backend}")
