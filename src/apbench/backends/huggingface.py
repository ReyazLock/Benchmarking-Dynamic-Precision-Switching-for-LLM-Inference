from __future__ import annotations

import math
import time

from ..config import ModelConfig
from ..measurements import FirstTokenTimer
from ..types import GenerationResult, GenerationSettings
from .base import Backend


class HuggingFaceBackend(Backend):
    """FP16/BF16 baseline for an apples-to-apples local comparison."""

    def __init__(self, config: ModelConfig):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install native dependencies with `pip install -e '.[native]'`") from exc
        self.torch = torch
        self.config = config
        dtype = torch.float16 if config.extra_options.get("dtype", "float16") == "float16" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        ).to(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=config.trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate(self, prompt, settings, precision_bits=None) -> GenerationResult:
        request_started = time.perf_counter()
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.config.device) for key, value in encoded.items()}
        def synchronize() -> None:
            if self.config.device.startswith("cuda"):
                self.torch.cuda.synchronize()

        synchronize()
        started = time.perf_counter()
        timer = FirstTokenTimer(request_started, synchronize)
        with self.torch.inference_mode():
            output = self.model.generate(
                **encoded,
                streamer=timer,
                max_new_tokens=settings.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
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
            metadata={"dtype": str(self.model.dtype)},
        )

    def perplexity(self, text: str, precision_bits: int | None = None) -> float | None:
        encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {key: value.to(self.config.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            output = self.model(**encoded, labels=encoded["input_ids"])
        return math.exp(float(output.loss.detach().cpu()))

    def runtime_info(self):
        if self.torch.cuda.is_available():
            return {
                "torch_allocated_bytes": self.torch.cuda.memory_allocated(),
                "torch_reserved_bytes": self.torch.cuda.memory_reserved(),
            }
        return {}

