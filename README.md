# AnyPrecision-Bench: TinyLlama Experiment Guide

This project uses Python to compare the FP16 baseline of TinyLlama-1.1B-Chat-v1.0 against Any-Precision packed model inference at w3 to w8. It measures answer quality, time to first token (TTFT), decoding speed (TPS), GPU memory, and performance across different input lengths. The native backend switches `precision` on the **same packed artifact**. The earlier experiments that used multiple GGUF files in Ollama are not this native Any-Precision test.

## Current Progress

- Uses the public model `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Quantization and packing are complete with seed precision 3 and parent precision 8, and the native CUDA extension and the packed model can both be loaded.
- Planned comparison: w3, w4, w5, w6, w7, w8 against FP16 of the same base checkpoint. Workload, TTFT, TPS, and prompt-sensitivity statistics have been added.
- For the GPU memory experiment, FP16 and the packed model run in **separate `apbench run` processes**. Keep the GPU idle, and record the actual model path, checkpoint revision, and run order.

> The repository's current `configs/anyprecision_example.yaml` still contains the generic `/models/base-llama` path. The two TinyLlama configs below are new example files; you must replace the packed model path and copy the data into the corresponding directory.

## Environment Setup (WSL)

Run from the Bench project root. For the Python 3.11 virtual environment, you can keep using the `.venv311` in which Any-Precision is already installed:

```bash
cd /path/to/Testing_Demo
source /path/to/any-precision-llm/.venv311/bin/activate
python -m pip install -e '.[dev,native]'
python -m pytest
python -c 'import torch, any_precision_ext; print(torch.cuda.is_available())'
nvidia-smi
```

The final Python command should print `True`. `pip install -e` does not compile the upstream CUDA kernel; if importing `any_precision_ext` fails, you must first finish installing that extension in the Any-Precision project. The native backend does not require starting Ollama.

Put the actual 30-question dataset in `datasets/` (you can copy it from its existing location). A formal QA set can later be placed separately as `formal_qa.jsonl`:

```bash
mkdir -p datasets configs results
cp /path/to/workload_qa_30.jsonl datasets/
wc -l datasets/workload_qa_30.jsonl
find /path/to/any-precision-llm -type d -name 'anyprec-*' -print
```

From the `find` results, choose a directory that has been packed and can be loaded with `AnyPrecisionForCausalLM.from_quantized`, not a gradient or LUT cache directory. Record its full WSL path. The earlier quantization used `--seed_precision 3 --parent_precision 8 --dataset c4 --seq_len 512 --num_examples 100 --random_state 42 --cpu_only`; `--cpu_only` belongs to the quantization process, and inference still uses CUDA.

## Config A: FP16 Baseline

Create `configs/tinyllama_fp16_workload.yaml`:

```yaml
run_name: tinyllama-fp16-workload
output_dir: ../results
repeats: 3
warmup_runs: 2
resource_sample_interval_s: 0.1
workload_ranges:
  - { id: small, min_tokens: 0, max_tokens: 128 }
  - { id: medium, min_tokens: 129, max_tokens: 512 }
  - { id: large, min_tokens: 513 }
generation:
  max_new_tokens: 64
  temperature: 0.0
  top_p: 1.0
  seed: 42
models:
  - id: tinyllama-fp16
    backend: huggingface
    model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    precision_bits: 16
    baseline: true
    device: cuda
    extra_options: { dtype: float16 }
datasets:
  - id: workload-qa-30
    path: ../datasets/workload_qa_30.jsonl
    task: qa
    metrics: [exact_match, contains_reference, token_f1]
    prompt_templates: ["{prompt}"]
```

If you already have a local FP16 model snapshot, you can change `model` to its absolute directory. Do not enter an Ollama tag.

## Config B: Native packed w8→w3

Create `configs/tinyllama_packed_workload.yaml`, **replacing `model` with the actual packed directory**:

```yaml
run_name: tinyllama-packed-workload
output_dir: ../results
repeats: 3
warmup_runs: 2
resource_sample_interval_s: 0.1
workload_ranges:
  - { id: small, min_tokens: 0, max_tokens: 128 }
  - { id: medium, min_tokens: 129, max_tokens: 512 }
  - { id: large, min_tokens: 513 }
generation:
  max_new_tokens: 64
  temperature: 0.0
  top_p: 1.0
  seed: 42
models:
  - id: tinyllama-anyprecision
    backend: anyprecision
    model: /absolute/wsl/path/to/tinyllama-packed
    precisions: [8, 7, 6, 5, 4, 3]
    device: cuda
    trust_remote_code: true
datasets:
  - id: workload-qa-30
    path: ../datasets/workload_qa_30.jsonl
    task: qa
    metrics: [exact_match, contains_reference, token_f1]
    prompt_templates: ["{prompt}"]
```

All six bit widths reuse the same packed model. The order here is 8→3; if you want to study the effect of run order, create a separate 3→8 config and save its results separately. The questions, templates, generation parameters, and workload boundaries of the two configs must be identical.

In the repository's current `src/apbench/metrics.py`, `rouge_l` divides by zero when the prediction and reference are both non-empty but share no tokens, so it is not enabled in the examples for now; add it back if your local version has fixed this. Each line of the JSONL data should have `id`, `prompt`, and `reference`.

## Running

```bash
apbench validate configs/tinyllama_fp16_workload.yaml
apbench validate configs/tinyllama_packed_workload.yaml
test -f datasets/workload_qa_30.jsonl

# Each command starts a separate process; wait for the previous one to finish before running the next
apbench run configs/tinyllama_fp16_workload.yaml
apbench run configs/tinyllama_packed_workload.yaml
```

`validate` checks the config format but does not verify the model files, data files, or GPU inference. For a pre-check, you can temporarily change `repeats` to 1 in both configs and add `max_samples: 2` under the dataset; restore them after confirming the results. With the formal parameters, one dataset of 30 questions × 1 template × 3 repeats gives 90 QA records for FP16 and 90 for each packed bit width, 540 in total. If `__warmup__` or `__model_load__` errors are present, investigate them before computing statistics.

Formal QA: copy the data to `datasets/formal_qa.jsonl`, copy the two YAML files, and change `run_name` and the dataset's `id` and `path`. For a prompt-sensitivity comparison, add the **same multiple templates** to both configs; with 3 templates, multiply the record count by 3. Using only one template does not produce a meaningful cross-template comparison.

## Output and Interpretation

Each run writes to `results/<run-name>-<UTC time>/`:

| File                             | Contents                                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------------------- |
| `raw_results.jsonl`              | Per-question output, metrics, tokens, timing, resources, and errors                         |
| `summary.csv`                    | Aggregated by model, precision, dataset, and template                                       |
| `workload_summary.csv`           | Further grouped by small / medium / large; includes sample count, TTFT, TPS, GPU memory     |
| `prompt_sensitivity.csv`         | Output variation across templates for the same question                                     |
| `baseline_comparison.csv`        | Differences computed automatically **only when the same run contains both FP16 and packed** |
| `run_metadata.json`, `report.md` | Config metadata and a human-readable report                                                 |

Because this guide splits FP16 and packed into two processes, **you will not automatically get a cross-process baseline comparison table**. To compare, align the raw records from the two results by the same `dataset_id`, `case_id`, `template_id`, and `repeat`, or compare the two `summary.csv` / `workload_summary.csv` files by corresponding groups; keep the run directories and their order.

- `exact_match` requires the normalized full answers to be identical. If the prediction is `Answer: 115` and the reference is `115`, the score is 0; `contains_reference` would be 1, but it may misjudge a wrong answer that happens to contain the correct number. Decide the primary accuracy criterion in advance and spot-check samples.
- `ttft_s` covers the time from the request to the first output token; even if the first generated token is EOS, the decoded answer may still be empty. `tps = (output_tokens - 1) / (wall_time_s - ttft_s)`; with fewer than two output tokens there is no TPS. `tokens_per_second` is a different rate computed over total generation time; do not mix them up.
- Workloads are classified by **actual input token count**: small is 0–128, medium is 129–512, and large starts at 513. The medium/large names of questions in the dataset do not guarantee the grouping; check `sample_count` and `mean_input_tokens` in `workload_summary.csv`.
- `peak_gpu_memory_used_mib` is the peak of whole-GPU memory sampled from `nvidia-smi` **during a single generation**; it does not include the peak during the entire model loading process, nor is it the memory exclusive to that precision. `torch_allocated_bytes` and `torch_reserved_bytes` use the PyTorch allocator's accounting.

### Memory Bias and Model Release

Currently `runner.py` calls `backend.close()` after each model finishes, but the existing backends do not implement explicit unloading. So if FP16 runs first and packed runs next in the same process, surviving references or the PyTorch cache may affect the subsequent whole-GPU memory readings. Merely adding `del backend; gc.collect(); torch.cuda.empty_cache()` does not guarantee that model tensors that are still referenced will be released, and it does not change the packed backend's design of **loading the same weights once to serve multiple bit widths**. Separate processes plus an idle-GPU baseline are better suited to comparing the resident footprint of FP16 versus packed; similar memory across w3–w8 cannot be used to claim that precision switching failed. To measure the exclusive loading cost of a specific bit width, you would need to design a model and corresponding backend that truly load only that bit width.

## Follow-up Research Checklist

1. Report separately: quality, empty-answer rate, truncation rate, TTFT/TPS, runtime GPU memory, packed file size, and the time and resources of the gradient/quantization/packing stages.
2. Record the base model revision, quantization config, CUDA/PyTorch/Transformers versions, and the GPU model tested; fix the data, templates, and generation settings.
3. Compare both the 8→3 and 3→8 orders; record idle GPU memory from `nvidia-smi` before each launch, and do not mistake whole-GPU peaks from different experiments for the model's inherent size.
4. After using multiple equivalent templates in formal QA, evaluate `prompt_sensitivity.csv` and the accuracy and empty-answer rate under different templates.

All paths in this guide are placeholders. Actual measurement results depend on your WSL machine, the packed model, and `raw_results.jsonl`.
