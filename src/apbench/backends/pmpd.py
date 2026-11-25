from __future__ import annotations

import gc
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import ModelConfig
from ..types import GenerationResult, GenerationSettings
from .base import Backend


class PMPDBackend(Backend):
    """PMPD's AnyPrecision model and static scheduler with measured greedy decoding.

    The upstream PMPD generate method performs an extra forward at the end and
    reports the *next* scheduled bit instead of the bit used for each token.
    This loop uses its actual model and scheduler while keeping Bench's token
    limit, first-token time, and precision accounting exact.
    """

    def __init__(self, config: ModelConfig):
        if config.device not in {"cuda", "cuda:0"}:
            raise ValueError("PMPDForCausalLM currently loads on CUDA; set device: cuda")
        try:
            import torch
            from pmpd import PMPDForCausalLM, Scheduler
        except ImportError as exc:
            raise RuntimeError(
                "PMPD imports failed. Install SamsungLabs/PMPD in the active WSL "
                "environment and check its AnyPrecision and AutoGPTQ imports."
            ) from exc

        self.torch = torch
        self.config = config
        self.high = int(config.precision_bits)
        self.low = int(config.extra_options.get("low_precision", min(config.precisions)))
        self.prefill = int(config.extra_options.get("prefill_precision", self.high))
        self.high_bit_steps = int(config.extra_options.get("high_bit_steps", 16))
        started = time.perf_counter()
        self.wrapper = PMPDForCausalLM(
            config.model, precisions=list(config.precisions), use_anyprec=True
        ).eval()
        self.wrapper.scheduler = Scheduler.get_scheduler(
            "naive", precisions=[self.high, self.low], high_bit_steps=self.high_bit_steps
        )
        self.tokenizer = self.wrapper.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.load_time_s = time.perf_counter() - started
        path = Path(config.model)
        self.artifact_size_bytes = (
            sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            if path.exists() else None
        )

    def generate(
        self, prompt: str, settings: GenerationSettings, precision_bits: int | None = None
    ) -> GenerationResult:
        if precision_bits != self.high:
            raise ValueError("PMPD precision_bits is the configured high precision")
        if settings.temperature != 0:
            raise ValueError("PMPD static adapter supports greedy decoding only")
        if settings.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

        torch = self.torch
        request_started = time.perf_counter()
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.config.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.config.device)
        if input_ids.shape[0] != 1:
            raise ValueError("PMPD benchmark expects one prompt per call")

        scheduler = self.wrapper.scheduler
        model = self.wrapper.model
        eos = self.tokenizer.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        generated = []
        trace = []
        ttft = None
        # current_bit = self.high
        current_bit = self.low if self.high_bit_steps == 0 else self.high
        logit_bit = self.prefill

        torch.cuda.synchronize()
        generation_started = time.perf_counter()
        scheduler.reset()
        try:
            with torch.inference_mode():
                outputs = model(
                    input_ids, attention_mask=attention_mask,
                    precision=self.prefill, use_cache=True,
                )
                for index in range(settings.max_new_tokens):
                    token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    token_id = int(token.item())
                    generated.append(token)
                    trace.append(logit_bit)
                    if ttft is None:
                        torch.cuda.synchronize()
                        ttft = time.perf_counter() - request_started
                    if token_id in eos_ids or index + 1 == settings.max_new_tokens:
                        break

                    if attention_mask is not None:
                        attention_mask = torch.cat(
                            [attention_mask, attention_mask.new_ones((1, 1))], dim=-1
                        )
                    outputs = model(
                        token, precision=current_bit,
                        past_key_values=outputs.past_key_values,
                        attention_mask=attention_mask, use_cache=True,
                    )
                    logit_bit = current_bit
                    # PMPD calls schedule after the first decode forward, with
                    # index=1. The selected precision is used on the next forward.
                    current_bit = scheduler.schedule(
                        index=index + 1, precision=current_bit,
                        past_key_values=outputs.past_key_values[-1],
                        logits=outputs.logits,
                    )
        finally:
            scheduler.reset()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - generation_started
        text = self.tokenizer.decode(torch.cat(generated, dim=-1)[0], skip_special_tokens=True)
        return GenerationResult(
            text=text,
            wall_time_s=time.perf_counter() - request_started,
            ttft_s=ttft,
            input_tokens=int(input_ids.shape[-1]),
            output_tokens=len(generated),
            generation_time_s=elapsed,
            metadata={
                "schedule": "pmpd_naive",
                "prefill_precision": self.prefill,
                "high_precision": self.high,
                "low_precision": self.low,
                "high_bit_steps": self.high_bit_steps,
                "token_precision_trace": trace,
                "token_precision_counts": dict(Counter(trace)),
            },
        )

    def runtime_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "native_load_time_s": self.load_time_s,
            "model_artifact_size_bytes": self.artifact_size_bytes,
        }
        if self.torch.cuda.is_available():
            info["torch_allocated_bytes"] = self.torch.cuda.memory_allocated()
            info["torch_reserved_bytes"] = self.torch.cuda.memory_reserved()
        return info

    def close(self) -> None:
        self.wrapper = None
        self.tokenizer = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
