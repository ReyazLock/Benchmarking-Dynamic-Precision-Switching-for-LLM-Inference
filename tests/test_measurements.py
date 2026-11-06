from pathlib import Path

import pytest

from apbench.config import DEFAULT_WORKLOAD_RANGES, WorkloadRange, load_config
from apbench.measurements import FirstTokenTimer, decode_tps, workload_for_tokens
from apbench.report import summarize_workloads


def test_workload_boundaries_and_two_ranges(tmp_path: Path):
    ranges = DEFAULT_WORKLOAD_RANGES
    assert [workload_for_tokens(value, ranges) for value in (0, 128, 129, 512, 513)] == [
        "small", "small", "medium", "medium", "large",
    ]
    assert workload_for_tokens(None, ranges) == "unknown"
    config = tmp_path / "two.yaml"
    config.write_text(
        "models:\n  - id: one\n    backend: ollama\n    model: example\n"
        "datasets:\n  - id: qa\n    path: qa.jsonl\n"
        "workload_ranges:\n"
        "  - {id: small, min_tokens: 0, max_tokens: 256}\n"
        "  - {id: large, min_tokens: 257}\n"
    )
    loaded = load_config(config)
    assert workload_for_tokens(257, loaded.workload_ranges) == "large"
    config.write_text(config.read_text().replace("min_tokens: 257", "min_tokens: 258"))
    with pytest.raises(ValueError, match="contiguous"):
        load_config(config)


def test_first_token_timer_and_decode_tps():
    calls = []
    timer = FirstTokenTimer(0, lambda: calls.append("sync"))
    timer.put("prompt")
    assert timer.ttft_s is None
    timer.put("first token")
    assert timer.ttft_s is not None
    timer.put("second token")
    timer.end()
    assert calls == ["sync"]
    assert decode_tps(4, 2.0, 0.5) == 2.0
    assert decode_tps(1, 2.0, 0.5) is None
    assert decode_tps(4, 0.5, 0.5) is None


def test_workload_summary_reports_untested_ranges():
    ranges = [vars(item) for item in DEFAULT_WORKLOAD_RANGES]
    metadata = {
        "models": [{"id": "one", "backend": "ollama", "precision_bits": 8, "precisions": []}],
        "datasets": [{"id": "qa", "task": "qa", "prompt_templates": ["{prompt}"]}],
        "workload_ranges": ranges,
    }
    records = [{
        "model_id": "one", "backend": "ollama", "precision_bits": 8,
        "dataset_id": "qa", "case_id": "a", "template_id": "template_0",
        "workload": "small", "input_tokens": 100, "output_tokens": 3,
        "ttft_s": 0.2, "wall_time_s": 0.6, "tps": 5.0,
    }]
    rows = {row["workload"]: row for row in summarize_workloads(records, metadata)}
    assert rows["small"]["sample_count"] == 1
    assert rows["small"]["mean_tps"] == 5.0
    assert rows["small"]["tps_sample_count"] == 1
    assert rows["medium"]["sample_count"] == 0
    assert rows["medium"]["tps_sample_count"] == 0
    assert rows["large"]["mean_ttft_s"] is None
