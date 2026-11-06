import json

from apbench.backends.ollama import OllamaBackend
from apbench.config import ModelConfig
from apbench.types import GenerationSettings


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload

    def iter_lines(self, chunk_size=1):
        assert chunk_size == 1
        for part in self.payload:
            yield json.dumps(part).encode()

    def close(self):
        pass


class FakeSession:
    def post(self, url, json, timeout, stream=False):
        assert stream and json["stream"]
        return FakeResponse(
            [
                {"response": "4", "done": False},
                {"response": "2", "done": False},
                {
                    "response": "", "done": True,
                    "prompt_eval_count": 5,
                    "eval_count": 2,
                    "load_duration": 1_000_000,
                    "prompt_eval_duration": 2_000_000,
                    "eval_duration": 4_000_000,
                    "total_duration": 7_000_000,
                    "done_reason": "stop",
                },
            ]
        )

    def get(self, url, timeout):
        return FakeResponse(
            {
                "models": [
                    {
                        "name": "test:int5",
                        "size": 100,
                        "size_vram": 80,
                        "context_length": 4096,
                        "details": {"quantization_level": "Q5_K_M"},
                    }
                ]
            }
        )

    def close(self):
        return None


def test_ollama_response_mapping():
    backend = OllamaBackend(
        ModelConfig(id="test", backend="ollama", model="test:int5", precision_bits=5)
    )
    backend.session = FakeSession()
    result = backend.generate("question", GenerationSettings(), 5)
    assert result.text == "42"
    assert result.output_tokens == 2
    assert result.ttft_s is not None and 0 <= result.ttft_s <= result.wall_time_s
    assert result.generation_time_s == 0.004
    assert backend.runtime_info()["reported_quantization"] == "Q5_K_M"

