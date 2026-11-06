from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from ..config import ModelConfig
from ..measurements import FirstTokenTimer
from ..types import GenerationResult, GenerationSettings
from .base import Backend


class AnyPrecisionBackend(Backend):
    """Native backend for SNU-ARC Any-Precision LLM.

    Imports are deliberately lazy so the Ollama-only workflow does not require
    PyTorch, Transformers, or the custom CUDA extension.
    """

    def __init__(self, config: ModelConfig):
        try:
            import torch
            from any_precision import AnyPrecisionForCausalLM
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Native backend dependencies are missing. Install the Any-Precision "
                "repository, its CUDA kernel, and `pip install -e '.[native]'`."
            ) from exc

        self.torch = torch
        self.config = config
        started = time.perf_counter()
        self.model = AnyPrecisionForCausalLM.from_quantized(
            config.model,
            trust_remote_code=config.trust_remote_code,
            fuse_layers=False,
            precisions=list(config.precisions),
        )
        self.model = self.model.to(config.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=config.trust_remote_code
        )
        self.load_time_s = time.perf_counter() - started
        model_path = Path(config.model)
        self.artifact_size_bytes = (
            sum(item.stat().st_size for item in model_path.rglob("*") if item.is_file())
            if model_path.exists()
            else None
        )
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(
        self,
        prompt: str,
        settings: GenerationSettings,
        precision_bits: int | None = None,
    ) -> GenerationResult:
        if precision_bits is None:
            raise ValueError("Any-Precision generation requires precision_bits")
        request_started = time.perf_counter()
        torch = self.torch
        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = torch.device(self.config.device)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        do_sample = settings.temperature > 0
        generator = torch.Generator(device=device).manual_seed(settings.seed)
        def synchronize() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        synchronize()
        started = time.perf_counter()
        timer = FirstTokenTimer(request_started, synchronize)
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                streamer=timer,
                precision=precision_bits,
                max_new_tokens=settings.max_new_tokens,
                do_sample=do_sample,
                temperature=settings.temperature if do_sample else None,
                top_p=settings.top_p if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                generator=generator if do_sample else None,
            )
        synchronize()
        elapsed = time.perf_counter() - started
        wall_time_s = time.perf_counter() - request_started
        input_count = int(encoded["input_ids"].shape[-1])
        generated = output[0, input_count:]
        return GenerationResult(
            text=self.tokenizer.decode(generated, skip_special_tokens=True),
            wall_time_s=wall_time_s,
            ttft_s=timer.ttft_s,
            input_tokens=input_count,
            output_tokens=int(generated.shape[-1]),
            generation_time_s=elapsed,
            metadata={"precision_bits": precision_bits},
        )

    def perplexity(self, text: str, precision_bits: int | None = None) -> float | None:
        if precision_bits is None:
            raise ValueError("Any-Precision perplexity requires precision_bits")
        torch = self.torch
        encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {key: value.to(self.config.device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = self.model(**encoded, labels=encoded["input_ids"], precision=precision_bits)
        return math.exp(float(output.loss.detach().cpu()))

    def runtime_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {"native_load_time_s": self.load_time_s}
        info["model_artifact_size_bytes"] = self.artifact_size_bytes
        if self.torch.cuda.is_available():
            info["torch_allocated_bytes"] = self.torch.cuda.memory_allocated()
            info["torch_reserved_bytes"] = self.torch.cuda.memory_reserved()
        return info
