from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42


@dataclass
class GenerationResult:
    text: str
    wall_time_s: float
    ttft_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    load_time_s: float | None = None
    prompt_eval_time_s: float | None = None
    generation_time_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    prompt: str = ""
    reference: str = ""
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

