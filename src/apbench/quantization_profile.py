from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .resources import ResourceSampler


def profile_quantization(
    repository: Path,
    model: str,
    output: Path,
    seed_precision: int = 3,
    parent_precision: int = 8,
    dataset: str = "c4",
    seq_len: int = 512,
    num_examples: int = 100,
    cpu_only: bool = False,
    sample_interval_s: float = 0.25,
) -> Path:
    quantize_script = repository / "quantize.py"
    if not quantize_script.is_file():
        raise FileNotFoundError(f"quantize.py not found in {repository}")
    output.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for mode in ("gradients", "quantize", "pack"):
        command = [
            "python",
            str(quantize_script),
            model,
            "--mode",
            mode,
            "--seed_precision",
            str(seed_precision),
            "--parent_precision",
            str(parent_precision),
            "--dataset",
            dataset,
            "--seq_len",
            str(seq_len),
            "--num_examples",
            str(num_examples),
            "--random_state",
            "42",
        ]
        if cpu_only:
            command.append("--cpu_only")
        print(f"[apbench] quantization phase: {mode}", flush=True)
        with ResourceSampler(sample_interval_s) as sampler:
            started = time.perf_counter()
            completed = subprocess.run(
                command,
                cwd=repository,
                capture_output=True,
                text=True,
                check=False,
            )
            elapsed = time.perf_counter() - started
        results.append(
            {
                "phase": mode,
                "elapsed_s": elapsed,
                "return_code": completed.returncode,
                "command": command,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
                **sampler.snapshot().to_dict(),
            }
        )
        if completed.returncode != 0:
            break
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repository": str(repository.resolve()),
        "model": model,
        "seed_precision": seed_precision,
        "parent_precision": parent_precision,
        "phases": results,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output
