from __future__ import annotations

import json
from pathlib import Path

from .types import BenchmarkCase


def load_jsonl(path: Path, max_samples: int | None = None) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = str(row.get("id", line_number))
            cases.append(
                BenchmarkCase(
                    case_id=case_id,
                    prompt=str(row.get("prompt", "")),
                    reference=str(row.get("reference", "")),
                    text=str(row.get("text", "")),
                    metadata=dict(row.get("metadata", {})),
                )
            )
            if max_samples is not None and len(cases) >= max_samples:
                break
    return cases
