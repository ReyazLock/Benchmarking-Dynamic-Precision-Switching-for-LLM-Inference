from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .types import GenerationSettings


SUPPORTED_BACKENDS = {"anyprecision", "huggingface", "pmpd"}


@dataclass(frozen=True)
class WorkloadRange:
    id: str
    min_tokens: int
    max_tokens: int | None = None


DEFAULT_WORKLOAD_RANGES = (
    WorkloadRange("small", 0, 128),
    WorkloadRange("medium", 129, 512),
    WorkloadRange("large", 513),
)


@dataclass(frozen=True)
class ModelConfig:
    id: str
    backend: str
    model: str
    precision_bits: int | None = None
    precisions: tuple[int, ...] = ()
    baseline: bool = False
    base_url: str = "http://localhost:11434"
    keep_alive: str = "5m"
    trust_remote_code: bool = False
    device: str = "cuda"
    extra_options: dict[str, Any] = field(default_factory=dict)

    def variants(self) -> tuple[int | None, ...]:
        if self.backend == "pmpd":
            # PMPD uses these precisions within one generation, not separate runs.
            return (self.precision_bits,)
        return self.precisions or (self.precision_bits,)


@dataclass(frozen=True)
class DatasetConfig:
    id: str
    path: Path
    task: str = "qa"
    prompt_templates: tuple[str, ...] = ("{prompt}",)
    metrics: tuple[str, ...] = ("exact_match", "token_f1")
    max_samples: int | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    run_name: str
    output_dir: Path
    repeats: int
    warmup_runs: int
    generation: GenerationSettings
    models: tuple[ModelConfig, ...]
    datasets: tuple[DatasetConfig, ...]
    resource_sample_interval_s: float = 0.1
    workload_ranges: tuple[WorkloadRange, ...] = DEFAULT_WORKLOAD_RANGES


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required key '{key}' in {context}")
    return mapping[key]


def load_config(path: str | Path) -> BenchmarkConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    base_dir = config_path.parent

    generation = GenerationSettings(**raw.get("generation", {}))
    models: list[ModelConfig] = []
    for item in raw.get("models", []):
        backend = str(_require(item, "backend", "models[]")).lower()
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}")
        precision = item.get("precision_bits")
        precisions = tuple(int(value) for value in item.get("precisions", []))
        all_bits = (*precisions, *((int(precision),) if precision is not None else ()))
        if any(value < 3 or value > 16 for value in all_bits):
            raise ValueError("precision values must be between 3 and 16")
        if backend == "anyprecision" and not precisions:
            raise ValueError("anyprecision models require a 'precisions' list")
        extra_options = dict(item.get("extra_options", {}))
        if backend == "pmpd":
            if not precisions or precision is None:
                raise ValueError("pmpd requires 'precisions' and high 'precision_bits'")
            if int(precision) not in precisions:
                raise ValueError("pmpd precision_bits must be in precisions")
            if extra_options.get("scheduler", "naive") != "naive":
                raise ValueError("pmpd currently supports only the naive static scheduler")
            low = int(extra_options.get("low_precision", min(precisions)))
            prefill = int(extra_options.get("prefill_precision", precision))
            if low not in precisions or prefill not in precisions or low >= int(precision):
                raise ValueError("pmpd requires available low < high and available prefill precision")
            steps = int(extra_options.get("high_bit_steps", 16))
            if steps < 0:
                raise ValueError("pmpd high_bit_steps must be >= 0")
            if generation.temperature != 0:
                raise ValueError("pmpd currently requires generation.temperature: 0")
        models.append(
            ModelConfig(
                id=str(_require(item, "id", "models[]")),
                backend=backend,
                model=str(_require(item, "model", "models[]")),
                precision_bits=int(precision) if precision is not None else None,
                precisions=precisions,
                baseline=bool(item.get("baseline", False)),
                base_url=str(item.get("base_url", "http://localhost:11434")).rstrip("/"),
                keep_alive=str(item.get("keep_alive", "5m")),
                trust_remote_code=bool(item.get("trust_remote_code", False)),
                device=str(item.get("device", "cuda")),
                extra_options=extra_options,
            )
        )

    datasets: list[DatasetConfig] = []
    for item in raw.get("datasets", []):
        dataset_path = Path(_require(item, "path", "datasets[]"))
        if not dataset_path.is_absolute():
            dataset_path = (base_dir / dataset_path).resolve()
        task = str(item.get("task", "qa"))
        if task not in {"qa", "language_modeling"}:
            raise ValueError(f"Unsupported dataset task: {task}")
        datasets.append(
            DatasetConfig(
                id=str(_require(item, "id", "datasets[]")),
                path=dataset_path,
                task=task,
                prompt_templates=tuple(item.get("prompt_templates", ["{prompt}"])),
                metrics=tuple(item.get("metrics", ["exact_match", "token_f1"])),
                max_samples=item.get("max_samples"),
            )
        )

    if not models:
        raise ValueError("At least one model is required")
    if not datasets:
        raise ValueError("At least one dataset is required")

    output_dir = Path(raw.get("output_dir", "results"))
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    repeats = int(raw.get("repeats", 3))
    warmup_runs = int(raw.get("warmup_runs", 1))
    if repeats < 1 or warmup_runs < 0:
        raise ValueError("repeats must be >= 1 and warmup_runs must be >= 0")
    configured_ranges = raw.get("workload_ranges")
    workload_ranges = (
        tuple(
            WorkloadRange(
                id=str(item["id"]),
                min_tokens=int(item["min_tokens"]),
                max_tokens=int(item["max_tokens"]) if item.get("max_tokens") is not None else None,
            )
            for item in configured_ranges
        )
        if configured_ranges is not None else DEFAULT_WORKLOAD_RANGES
    )
    if len(workload_ranges) not in (2, 3):
        raise ValueError("workload_ranges must contain 2 or 3 ranges")
    if len({item.id for item in workload_ranges}) != len(workload_ranges):
        raise ValueError("workload range IDs must be unique")
    for index, item in enumerate(workload_ranges):
        if not item.id or item.min_tokens < 0:
            raise ValueError("workload range IDs must be nonempty and min_tokens >= 0")
        if item.max_tokens is not None and item.max_tokens < item.min_tokens:
            raise ValueError("max_tokens must be >= min_tokens")
        if index == 0 and item.min_tokens != 0:
            raise ValueError("workload ranges must start at 0")
        if index < len(workload_ranges) - 1 and item.max_tokens is None:
            raise ValueError("only the last workload range can be unbounded")
        if index > 0 and item.min_tokens != workload_ranges[index - 1].max_tokens + 1:
            raise ValueError("workload ranges must be contiguous and ordered")
    if workload_ranges[-1].max_tokens is not None:
        raise ValueError("the last workload range must be unbounded")
    return BenchmarkConfig(
        run_name=str(raw.get("run_name", config_path.stem)),
        output_dir=output_dir,
        repeats=repeats,
        warmup_runs=warmup_runs,
        generation=generation,
        models=tuple(models),
        datasets=tuple(datasets),
        resource_sample_interval_s=float(raw.get("resource_sample_interval_s", 0.1)),
        workload_ranges=workload_ranges,
    )
