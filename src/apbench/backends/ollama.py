from __future__ import annotations

import json
import time
from typing import Any

import requests

from ..config import ModelConfig
from ..types import GenerationResult, GenerationSettings
from .base import Backend


def _ns_to_s(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000_000_000


class OllamaBackend(Backend):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.session = requests.Session()

    def generate(
        self,
        prompt: str,
        settings: GenerationSettings,
        precision_bits: int | None = None,
    ) -> GenerationResult:
        options = {
            "num_predict": settings.max_new_tokens,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "seed": settings.seed,
            **self.config.extra_options,
        }
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
            "keep_alive": self.config.keep_alive,
            "options": options,
        }
        started = time.perf_counter()
        response = self.session.post(
            f"{self.config.base_url}/api/generate", json=payload, stream=True, timeout=None
        )
        chunks: list[str] = []
        ttft_s: float | None = None
        data: dict[str, Any] = {}
        try:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=1):
                if not line:
                    continue
                data = json.loads(line)
                if data.get("error"):
                    raise RuntimeError(f"Ollama generation failed: {data['error']}")
                piece = str(data.get("response") or "")
                if piece:
                    if ttft_s is None:
                        ttft_s = time.perf_counter() - started
                    chunks.append(piece)
                if data.get("done"):
                    break
        finally:
            response.close()
        if not data.get("done"):
            raise RuntimeError("Ollama stream ended without a final done event")
        wall_time_s = time.perf_counter() - started
        return GenerationResult(
            text="".join(chunks),
            wall_time_s=wall_time_s,
            ttft_s=ttft_s,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            load_time_s=_ns_to_s(data.get("load_duration")),
            prompt_eval_time_s=_ns_to_s(data.get("prompt_eval_duration")),
            generation_time_s=_ns_to_s(data.get("eval_duration")),
            metadata={
                "done_reason": data.get("done_reason"),
                "ollama_total_time_s": _ns_to_s(data.get("total_duration")),
                "configured_precision_bits": precision_bits,
            },
        )

    def runtime_info(self) -> dict[str, Any]:
        try:
            response = self.session.get(f"{self.config.base_url}/api/ps", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])
        except requests.RequestException as exc:
            return {"runtime_info_error": str(exc)}
        target = next(
            (
                item
                for item in models
                if item.get("name") == self.config.model
                or item.get("model") == self.config.model
            ),
            None,
        )
        if target is None:
            return {"runtime_model_loaded": False}
        details = target.get("details", {})
        return {
            "runtime_model_loaded": True,
            "model_size_bytes": target.get("size"),
            "model_vram_bytes": target.get("size_vram"),
            "context_length": target.get("context_length"),
            "reported_quantization": details.get("quantization_level"),
            "parameter_size": details.get("parameter_size"),
            "format": details.get("format"),
        }

    def close(self) -> None:
        try:
            self.session.post(
                f"{self.config.base_url}/api/generate",
                json={"model": self.config.model, "keep_alive": 0},
                timeout=30,
            )
        except requests.RequestException:
            pass
        self.session.close()
