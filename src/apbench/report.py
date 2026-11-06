from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["model_id"],
            record["backend"],
            record.get("precision_bits"),
            record["dataset_id"],
            record.get("template_id", "lm"),
        )
        groups[key].append(record)

    rows: list[dict[str, Any]] = []
    for key, items in groups.items():
        row: dict[str, Any] = dict(
            zip(("model_id", "backend", "precision_bits", "dataset_id", "template_id"), key)
        )
        numeric_fields = {
            field
            for item in items
            for field, value in item.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        for field in sorted(numeric_fields):
            values = [float(item[field]) for item in items if item.get(field) is not None]
            if values:
                row[f"mean_{field}"] = _mean(values)
        latencies = [float(item["wall_time_s"]) for item in items if item.get("wall_time_s") is not None]
        row["p50_wall_time_s"] = _percentile(latencies, 0.50)
        row["p95_wall_time_s"] = _percentile(latencies, 0.95)
        ttfts = [float(item["ttft_s"]) for item in items if item.get("ttft_s") is not None]
        row["p50_ttft_s"] = _percentile(ttfts, 0.50)
        row["p95_ttft_s"] = _percentile(ttfts, 0.95)
        predictions = defaultdict(list)
        for item in items:
            if "prediction" in item:
                predictions[item["case_id"]].append(item["prediction"])
        stability = [len(set(outputs)) / len(outputs) for outputs in predictions.values() if outputs]
        row["mean_unique_output_ratio"] = _mean(stability)
        row["sample_count"] = len(items)
        rows.append(row)
    return rows


def summarize_workloads(
    records: list[dict[str, Any]], metadata: dict[str, Any]
) -> list[dict[str, Any]]:
    """Include zero-sample rows so untested workload ranges are visible."""
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    labels = {item["id"]: item for item in metadata.get("workload_ranges", [])}
    for record in records:
        if record.get("dataset_id", "").startswith("__") or record.get("template_id") == "lm":
            continue
        key = (
            record["model_id"], record["backend"], record.get("precision_bits"),
            record["dataset_id"], record["template_id"], record.get("workload", "unknown"),
        )
        groups[key].append(record)
    for model in metadata.get("models", []):
        bits = model.get("precisions") or [model.get("precision_bits")]
        for dataset in metadata.get("datasets", []):
            if dataset.get("task") != "qa":
                continue
            for index, _ in enumerate(dataset.get("prompt_templates", ["{prompt}"])):
                for bit in bits:
                    for label in labels:
                        key = (model["id"], model["backend"], bit, dataset["id"], f"template_{index}", label)
                        groups.setdefault(key, [])

    rows = []
    for (model_id, backend, bit, dataset_id, template_id, workload), items in groups.items():
        measured = [item for item in items if "error" not in item]
        ranges = labels.get(workload, {})
        row = {
            "model_id": model_id,
            "backend": backend,
            "precision_bits": bit,
            "dataset_id": dataset_id,
            "template_id": template_id,
            "workload": workload,
            "min_tokens": ranges.get("min_tokens"),
            "max_tokens": ranges.get("max_tokens"),
            "sample_count": len(items),
            "unique_case_count": len({item["case_id"] for item in items}),
            "error_count": len(items) - len(measured),
            "ttft_sample_count": sum(item.get("ttft_s") is not None for item in measured),
            "tps_sample_count": sum(item.get("tps") is not None for item in measured),
        }
        for field in (
            "ttft_s", "tps", "wall_time_s", "input_tokens", "output_tokens",
            "exact_match", "contains_reference", "token_f1", "rouge_l",
            "peak_gpu_memory_used_mib",
        ):
            values = [float(item[field]) for item in measured if item.get(field) is not None]
            row[f"mean_{field}"] = _mean(values)
        for field in ("ttft_s", "wall_time_s"):
            values = [float(item[field]) for item in measured if item.get(field) is not None]
            row[f"p50_{field}"] = _percentile(values, 0.50)
            row[f"p95_{field}"] = _percentile(values, 0.95)
        rows.append(row)
    return rows


def write_outputs(run_dir: Path, records: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "raw_results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize(records)
    fields = sorted({field for row in summary for field in row})
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    workload_rows = summarize_workloads(records, metadata)
    workload_fields = [
        "model_id", "backend", "precision_bits", "dataset_id", "template_id", "workload",
        "min_tokens", "max_tokens", "sample_count", "unique_case_count", "error_count",
        "ttft_sample_count", "tps_sample_count",
        "mean_input_tokens", "mean_output_tokens", "mean_ttft_s", "p50_ttft_s",
        "p95_ttft_s", "mean_tps", "mean_wall_time_s", "p50_wall_time_s",
        "p95_wall_time_s", "mean_exact_match", "mean_contains_reference",
        "mean_token_f1", "mean_rouge_l", "mean_peak_gpu_memory_used_mib",
    ]
    with (run_dir / "workload_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=workload_fields)
        writer.writeheader()
        writer.writerows(workload_rows)
    _write_baseline_comparison(run_dir / "baseline_comparison.csv", summary, metadata)
    with (run_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    _write_prompt_sensitivity(run_dir / "prompt_sensitivity.csv", records)
    _write_markdown(run_dir / "report.md", summary, metadata)


def _write_baseline_comparison(
    path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> None:
    baseline_ids = {
        model["id"] for model in metadata.get("models", []) if model.get("baseline")
    }
    baselines: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row["model_id"] in baseline_ids:
            baselines[(row["dataset_id"], row["template_id"])] = row
    comparisons: list[dict[str, Any]] = []
    compare_fields = (
        "mean_exact_match",
        "mean_token_f1",
        "mean_rouge_l",
        "mean_perplexity",
        "mean_ttft_s",
        "p50_ttft_s",
        "mean_tps",
        "p50_wall_time_s",
        "mean_tokens_per_second",
        "mean_model_vram_bytes",
        "mean_peak_gpu_memory_used_mib",
        "mean_peak_ollama_rss_bytes",
    )
    for row in rows:
        baseline = baselines.get((row["dataset_id"], row["template_id"]))
        if baseline is None or row["model_id"] in baseline_ids:
            continue
        result = {
            "model_id": row["model_id"],
            "precision_bits": row.get("precision_bits"),
            "dataset_id": row["dataset_id"],
            "template_id": row["template_id"],
            "baseline_model_id": baseline["model_id"],
        }
        for field in compare_fields:
            value = row.get(field)
            base = baseline.get(field)
            if value is not None and base is not None:
                result[f"{field}_delta"] = value - base
                if base != 0:
                    result[f"{field}_change_pct"] = (value - base) / base * 100
        comparisons.append(result)
    fields = sorted({field for row in comparisons for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            handle.write("model_id,precision_bits,dataset_id,template_id,baseline_model_id\n")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparisons)


def _write_prompt_sensitivity(path: Path, records: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        if "prediction" not in item:
            continue
        key = (
            item["model_id"],
            item.get("precision_bits"),
            item["dataset_id"],
            item["case_id"],
            item.get("repeat", 0),
        )
        groups[key].append(item)
    rows: list[dict[str, Any]] = []
    for key, items in groups.items():
        if len({item.get("template_id") for item in items}) < 2:
            continue
        row = dict(zip(("model_id", "precision_bits", "dataset_id", "case_id", "repeat"), key))
        normalized_outputs = {" ".join(item["prediction"].lower().split()) for item in items}
        row["template_count"] = len(items)
        row["unique_output_ratio"] = len(normalized_outputs) / len(items)
        metric_names = {
            field
            for item in items
            for field in item
            if field in {"exact_match", "contains_reference", "token_f1", "rouge_l"}
        }
        for metric in sorted(metric_names):
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            row[f"{metric}_range"] = max(values) - min(values) if values else None
        rows.append(row)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            handle.write("model_id,precision_bits,dataset_id,case_id,repeat,template_count,unique_output_ratio\n")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_markdown(path: Path, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    columns = [
        "model_id",
        "precision_bits",
        "dataset_id",
        "template_id",
        "mean_exact_match",
        "mean_token_f1",
        "mean_perplexity",
        "p50_ttft_s",
        "mean_tps",
        "p50_wall_time_s",
        "mean_tokens_per_second",
        "mean_peak_gpu_memory_used_mib",
        "mean_model_vram_bytes",
    ]
    lines = [
        f"# Benchmark report: {metadata.get('run_name', '')}",
        "",
        f"Generated at: {metadata.get('created_at', '')}",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- Compare quality against the configured FP16 or highest-bit baseline; do not interpret bit-width alone as quality.",
            "- Whole-device GPU samples include other processes. Use an otherwise idle machine.",
            "- `mean_unique_output_ratio` near 1 means repeated runs were unstable; near 0 is impossible, and 1/repeats is fully stable.",
            "- Ollama's reported quantization string is authoritative for its model; `precision_bits` is only the experiment label.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
