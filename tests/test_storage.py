import json

from apbench.storage import compare_storage


def test_storage_comparison(tmp_path):
    packed = tmp_path / "packed"
    packed.mkdir()
    (packed / "weights.bin").write_bytes(b"12345")
    first = tmp_path / "int5.bin"
    second = tmp_path / "int8.bin"
    first.write_bytes(b"1234")
    second.write_bytes(b"123456")
    output = tmp_path / "result.json"
    compare_storage(packed, [first, second], output)
    result = json.loads(output.read_text())
    assert result["packed_size_bytes"] == 5
    assert result["independent_total_bytes"] == 10
    assert result["savings_pct"] == 50.0
