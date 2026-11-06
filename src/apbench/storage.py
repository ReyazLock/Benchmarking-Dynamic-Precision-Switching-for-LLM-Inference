from __future__ import annotations

import json
from pathlib import Path


def path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    raise FileNotFoundError(path)


def compare_storage(packed: Path, independent: list[Path], output: Path) -> Path:
    packed_size = path_size(packed)
    independent_sizes = {str(path): path_size(path) for path in independent}
    independent_total = sum(independent_sizes.values())
    savings = independent_total - packed_size
    payload = {
        "packed_path": str(packed.resolve()),
        "packed_size_bytes": packed_size,
        "independent_sizes_bytes": independent_sizes,
        "independent_total_bytes": independent_total,
        "savings_bytes": savings,
        "savings_pct": (savings / independent_total * 100) if independent_total else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output
