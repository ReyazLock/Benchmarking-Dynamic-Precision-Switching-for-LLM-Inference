from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from apbench.config import ModelConfig, load_config
from apbench.types import GenerationSettings


def test_pmpd_config_is_one_scheduled_run():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "tinyllama_pmpd_example.yaml")
    model = config.models[0]
    assert model.variants() == (8,)
    assert model.precisions == (3, 4, 5, 6, 7, 8)
    assert config.datasets[0].path.exists()


def test_pmpd_rejects_missing_low_precision(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "models:\n  - {id: bad, backend: pmpd, model: /tmp/model, "
        "precision_bits: 8, precisions: [3, 8], "
        "extra_options: {low_precision: 5}}\n"
        "datasets:\n  - {id: qa, path: qa.jsonl}\n"
    )
    with pytest.raises(ValueError, match="available low"):
        load_config(path)


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)
        self.shape = (1, len(values))

    def to(self, device):
        return self

    def new_ones(self, shape):
        return FakeTensor([1] * shape[-1])

    def __getitem__(self, key):
        return self if key == 0 else FakeTensor(self.values[key])


class FakeToken(FakeTensor):
    def __init__(self, value):
        super().__init__([value])

    def item(self):
        return self.values[0]


class FakeLogits:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        return self

    def argmax(self, **kwargs):
        return FakeToken(self.value)


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = None

    def __call__(self, prompt, return_tensors):
        return {"input_ids": FakeTensor([7, 9]), "attention_mask": FakeTensor([1, 1])}

    def decode(self, tokens, skip_special_tokens):
        return " ".join(str(value) for value in tokens.values if value != 2)


class FakeScheduler:
    def __init__(self, high_bit_steps):
        self.high_bit_steps = high_bit_steps
        self.indices = []

    def reset(self):
        pass

    def schedule(self, **kwargs):
        self.indices.append(kwargs["index"])
        return 8 if kwargs["index"] < self.high_bit_steps else 3


class FakeModel:
    def __init__(self):
        self.calls = []
        self.tokens = [11, 12, 13, 2]

    def __call__(self, input_ids, precision, **kwargs):
        self.calls.append(precision)
        position = len(self.calls) - 1
        return SimpleNamespace(
            logits=FakeLogits(self.tokens[position]),
            past_key_values=[("k", "v")],
        )


class FakeWrapper:
    def __init__(self, path, precisions, use_anyprec):
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()
        self.scheduler = None

    def eval(self):
        return self


def test_pmpd_uses_static_schedule_and_counts_actual_token_bits(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            synchronize=lambda: None,
            is_available=lambda: False,
        ),
        inference_mode=lambda: __import__("contextlib").nullcontext(),
        cat=lambda tensors, dim: FakeTensor(
            [item for tensor in tensors for item in tensor.values]
        ),
    )
    schedulers = []

    class SchedulerFactory:
        @staticmethod
        def get_scheduler(name, **kwargs):
            assert name == "naive"
            assert kwargs["precisions"] == [8, 3]
            scheduler = FakeScheduler(kwargs["high_bit_steps"])
            schedulers.append(scheduler)
            return scheduler

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(
        sys.modules, "pmpd",
        SimpleNamespace(PMPDForCausalLM=FakeWrapper, Scheduler=SchedulerFactory),
    )
    from apbench.backends.pmpd import PMPDBackend

    config = ModelConfig(
        id="fake", backend="pmpd", model="/nonexistent/model",
        precision_bits=8, precisions=(3, 4, 5, 6, 7, 8),
        extra_options={"low_precision": 3, "high_bit_steps": 2},
    )
    backend = PMPDBackend(config)
    result = backend.generate("prompt", GenerationSettings(max_new_tokens=4), 8)
    assert result.output_tokens == 4
    assert result.text == "11 12 13"
    assert result.metadata["token_precision_trace"] == [8, 8, 8, 3]
    assert backend.wrapper.model.calls == [8, 8, 8, 3]
    assert schedulers[0].indices == [1, 2, 3]
    assert result.ttft_s is not None

    backend.wrapper.model.calls.clear()
    limited = backend.generate("prompt", GenerationSettings(max_new_tokens=2), 8)
    assert limited.output_tokens == 2
    assert backend.wrapper.model.calls == [8, 8]
