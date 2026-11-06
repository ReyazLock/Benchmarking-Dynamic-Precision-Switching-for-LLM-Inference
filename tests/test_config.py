from pathlib import Path

from apbench.config import load_config


def test_example_config_is_valid():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "ollama_example.yaml")

    assert len(config.models) == 6

    precisions = {model.precision_bits for model in config.models}
    assert precisions == {3, 4, 5, 6, 8, 16}

    baselines = [model for model in config.models if model.baseline]
    assert len(baselines) == 1
    assert baselines[0].precision_bits == 16

    assert config.datasets[0].path.is_file()