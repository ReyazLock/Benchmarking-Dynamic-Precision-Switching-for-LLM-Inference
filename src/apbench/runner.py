from __future__ import annotations

import platform
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends import create_backend
from .config import BenchmarkConfig, DatasetConfig, ModelConfig
from .datasets import load_jsonl
from .metrics import evaluate
from .measurements import decode_tps, workload_for_tokens
from .report import write_outputs
from .resources import ResourceSampler


def _run_dir(config: BenchmarkConfig) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return config.output_dir / f"{config.run_name}-{timestamp}"


def _metadata(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "generation": asdict(config.generation),
        "models": [asdict(model) for model in config.models],
        "workload_ranges": [asdict(item) for item in config.workload_ranges],
        "datasets": [
            {**asdict(dataset), "path": str(dataset.path)} for dataset in config.datasets
        ],
    }


def _qa_records(
    backend,
    model: ModelConfig,
    precision: int | None,
    dataset: DatasetConfig,
    config: BenchmarkConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cases = load_jsonl(dataset.path, dataset.max_samples)
    for template_index, template in enumerate(dataset.prompt_templates):
        template_id = f"template_{template_index}"
        for case in cases:
            values = {"prompt": case.prompt, "reference": case.reference, **case.metadata}
            prompt = template.format(**values)
            for repeat in range(config.repeats):
                settings = replace(config.generation, seed=config.generation.seed + repeat)
                with ResourceSampler(config.resource_sample_interval_s) as sampler:
                    started = time.perf_counter()
                    try:
                        result = backend.generate(prompt, settings, precision)
                        error = None
                    except Exception as exc:  # preserve long-running experiments
                        result = None
                        error = f"{type(exc).__name__}: {exc}"
                    measured_wall = time.perf_counter() - started
                resource = sampler.snapshot().to_dict()
                base = {
                    "model_id": model.id,
                    "backend": model.backend,
                    "model": model.model,
                    "precision_bits": precision,
                    "baseline": model.baseline,
                    "dataset_id": dataset.id,
                    "case_id": case.case_id,
                    "template_id": template_id,
                    "repeat": repeat,
                    **resource,
                }
                if error is not None:
                    records.append({**base, "error": error, "wall_time_s": measured_wall, "workload": "unknown"})
                    continue
                metric_values = evaluate(result.text, case.reference, dataset.metrics)
                generation_time = result.generation_time_s or result.wall_time_s
                tokens_per_second = (
                    result.output_tokens / generation_time
                    if result.output_tokens is not None and generation_time > 0
                    else None
                )
                records.append(
                    {
                        **base,
                        **result.to_dict(),
                        **result.metadata,
                        **backend.runtime_info(),
                        **metric_values,
                        "tokens_per_second": tokens_per_second,
                        "tps": decode_tps(result.output_tokens, result.wall_time_s, result.ttft_s),
                        "workload": workload_for_tokens(result.input_tokens, config.workload_ranges),
                        "prediction": result.text,
                        "reference": case.reference,
                        "rendered_prompt": prompt,
                    }
                )
    return records


def _lm_records(
    backend,
    model: ModelConfig,
    precision: int | None,
    dataset: DatasetConfig,
    config: BenchmarkConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cases = load_jsonl(dataset.path, dataset.max_samples)
    for case in cases:
        if not case.text:
            raise ValueError(f"Language-modeling case {case.case_id} has no 'text' field")
        for repeat in range(config.repeats):
            with ResourceSampler(config.resource_sample_interval_s) as sampler:
                started = time.perf_counter()
                try:
                    perplexity = backend.perplexity(case.text, precision)
                    error = None
                    if perplexity is None:
                        error = f"Backend '{model.backend}' does not expose input-token logits"
                except Exception as exc:
                    perplexity = None
                    error = f"{type(exc).__name__}: {exc}"
                elapsed = time.perf_counter() - started
            record = {
                "model_id": model.id,
                "backend": model.backend,
                "model": model.model,
                "precision_bits": precision,
                "baseline": model.baseline,
                "dataset_id": dataset.id,
                "case_id": case.case_id,
                "template_id": "lm",
                "repeat": repeat,
                "perplexity": perplexity,
                "wall_time_s": elapsed,
                **sampler.snapshot().to_dict(),
                **backend.runtime_info(),
            }
            if error:
                record["error"] = error
            records.append(record)
    return records


def run_benchmark(config: BenchmarkConfig) -> Path:
    run_dir = _run_dir(config)
    records: list[dict[str, Any]] = []
    metadata = _metadata(config)
    run_dir.mkdir(parents=True, exist_ok=False)

    for model in config.models:
        print(f"[apbench] loading {model.id} ({model.backend})", flush=True)
        try:
            backend = create_backend(model)
        except Exception as exc:
            records.append(
                {
                    "model_id": model.id,
                    "backend": model.backend,
                    "model": model.model,
                    "precision_bits": model.precision_bits,
                    "dataset_id": "__model_load__",
                    "case_id": "__model_load__",
                    "template_id": "none",
                    "repeat": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        try:
            for precision in model.variants():
                print(f"[apbench] {model.id}: precision={precision}", flush=True)
                warmup_case = load_jsonl(config.datasets[0].path, 1)[0]
                if config.datasets[0].task == "qa":
                    warmup_prompt = config.datasets[0].prompt_templates[0].format(
                        prompt=warmup_case.prompt,
                        reference=warmup_case.reference,
                        **warmup_case.metadata,
                    )
                    try:
                        for _ in range(config.warmup_runs):
                            backend.generate(warmup_prompt, config.generation, precision)
                    except Exception as exc:
                        records.append(
                            {
                                "model_id": model.id,
                                "backend": model.backend,
                                "model": model.model,
                                "precision_bits": precision,
                                "dataset_id": "__warmup__",
                                "case_id": "__warmup__",
                                "template_id": "none",
                                "repeat": 0,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        write_outputs(run_dir, records, metadata)
                        continue
                for dataset in config.datasets:
                    if dataset.task == "qa":
                        records.extend(_qa_records(backend, model, precision, dataset, config))
                    else:
                        records.extend(_lm_records(backend, model, precision, dataset, config))
                write_outputs(run_dir, records, metadata)
        finally:
            backend.close()
            del backend
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    write_outputs(run_dir, records, metadata)
    return run_dir
