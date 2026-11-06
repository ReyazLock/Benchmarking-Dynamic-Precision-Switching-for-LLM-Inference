from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .quantization_profile import profile_quantization
from .runner import run_benchmark
from .storage import compare_storage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="apbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run an inference benchmark")
    run.add_argument("config", type=Path)

    validate = subparsers.add_parser("validate", help="validate a benchmark config")
    validate.add_argument("config", type=Path)

    profile = subparsers.add_parser(
        "profile-quantization", help="profile gradient, quantize, and pack phases"
    )
    profile.add_argument("--repo", required=True, type=Path)
    profile.add_argument("--model", required=True)
    profile.add_argument("--output", default="results/quantization_profile.json", type=Path)
    profile.add_argument("--seed-precision", default=3, type=int)
    profile.add_argument("--parent-precision", default=8, type=int)
    profile.add_argument("--dataset", default="c4")
    profile.add_argument("--seq-len", default=512, type=int)
    profile.add_argument("--num-examples", default=100, type=int)
    profile.add_argument("--cpu-only", action="store_true")

    storage = subparsers.add_parser(
        "compare-storage", help="compare one packed overlay with independent artifacts"
    )
    storage.add_argument("--packed", required=True, type=Path)
    storage.add_argument("--independent", required=True, nargs="+", type=Path)
    storage.add_argument("--output", default="results/storage_comparison.json", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        config = load_config(args.config)
        print(
            f"valid: {len(config.models)} model(s), "
            f"{len(config.datasets)} dataset(s), output={config.output_dir}"
        )
        return
    if args.command == "run":
        output = run_benchmark(load_config(args.config))
        print(f"[apbench] results: {output}")
        return
    if args.command == "profile-quantization":
        output = profile_quantization(
            repository=args.repo,
            model=args.model,
            output=args.output,
            seed_precision=args.seed_precision,
            parent_precision=args.parent_precision,
            dataset=args.dataset,
            seq_len=args.seq_len,
            num_examples=args.num_examples,
            cpu_only=args.cpu_only,
        )
        print(f"[apbench] profile: {output}")
        return
    if args.command == "compare-storage":
        output = compare_storage(args.packed, args.independent, args.output)
        print(f"[apbench] storage comparison: {output}")


if __name__ == "__main__":
    main()
