from __future__ import annotations

import time
from collections.abc import Callable

from .config import WorkloadRange


def workload_for_tokens(
    input_tokens: int | None, ranges: tuple[WorkloadRange, ...]
) -> str:
    if input_tokens is None:
        return "unknown"
    for item in ranges:
        if input_tokens >= item.min_tokens and (
            item.max_tokens is None or input_tokens <= item.max_tokens
        ):
            return item.id
    return "out_of_range"


def decode_tps(output_tokens: int | None, wall_time_s: float, ttft_s: float | None) -> float | None:
    """Tokens after the first / elapsed time after the first token.

    A one-token response has no measured inter-token interval, so its TPS is undefined.
    """
    if output_tokens is None or output_tokens < 2 or ttft_s is None:
        return None
    decode_time_s = wall_time_s - ttft_s
    return (output_tokens - 1) / decode_time_s if decode_time_s > 0 else None


class FirstTokenTimer:
    """Transformers-compatible streamer that timestamps the first generated token.

    Decoder-only models first send the complete prompt to ``put``. It is not a
    generated token and must be ignored. Synchronization avoids measuring only
    CUDA kernel dispatch rather than token completion.
    """

    def __init__(self, started: float, synchronize: Callable[[], None]):
        self.started = started
        self.synchronize = synchronize
        self.skip_prompt = True
        self.ttft_s: float | None = None

    def put(self, value: object) -> None:
        if self.skip_prompt:
            self.skip_prompt = False
        elif self.ttft_s is None:
            self.synchronize()
            self.ttft_s = time.perf_counter() - self.started

    def end(self) -> None:
        pass
