from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..types import GenerationResult, GenerationSettings


class Backend(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        settings: GenerationSettings,
        precision_bits: int | None = None,
    ) -> GenerationResult:
        raise NotImplementedError

    def perplexity(self, text: str, precision_bits: int | None = None) -> float | None:
        return None

    def runtime_info(self) -> dict[str, Any]:
        return {}

    def close(self) -> None:
        return None

