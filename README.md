# AnyPrecision-Bench：TinyLlama 实验指南

本项目使用 Python 比较 TinyLlama-1.1B-Chat-v1.0 的 FP16 基线与 Any-Precision packed 模型的 w3～w8 推理，测量回答质量、首 token 延迟（TTFT）、解码速度（TPS）、显存和不同输入长度的表现。原生后端在**同一个 packed artifact**上切换 `precision`；此前在 Ollama 中使用多个 GGUF 文件的实验不是这项原生 Any-Precision 测试。

## 当前进度

- 使用公开模型 `TinyLlama/TinyLlama-1.1B-Chat-v1.0`，已完成 seed precision 3、parent precision 8 的量化和 pack，并已能加载原生 CUDA 扩展及 packed 模型。
- 计划比较 w3、w4、w5、w6、w7、w8 与同一基础 checkpoint 的 FP16；已加入 workload、TTFT、TPS 和 prompt 敏感性统计。
- 显存实验将 FP16 与 packed 模型放在**不同的 `apbench run` 进程**中。请保持 GPU 空闲，并记录实际模型路径、checkpoint 修订号及运行顺序。

> 当前仓库的 `configs/anyprecision_example.yaml` 仍包含通用 `/models/base-llama` 路径。下文两份 TinyLlama 配置是新文件示例；必须替换 packed 模型路径，并将数据复制到相应目录。

## 环境准备（WSL）

从 Bench 项目根目录执行；Python 3.11 虚拟环境可继续使用已经安装 Any-Precision 的 `.venv311`：

```bash
cd /path/to/Testing_Demo
source /path/to/any-precision-llm/.venv311/bin/activate
python -m pip install -e '.[dev,native]'
python -m pytest
python -c 'import torch, any_precision_ext; print(torch.cuda.is_available())'
nvidia-smi
```

最后的 Python 命令应输出 `True`。`pip install -e` 不会编译上游的 CUDA kernel；若 `any_precision_ext` 导入失败，需先在 Any-Precision 项目中完成该扩展安装。原生后端不需要启动 Ollama。

把实际的 30 题数据放在 `datasets/`（可从现有位置复制）；后续正式 QA 可另放 `formal_qa.jsonl`：

```bash
mkdir -p datasets configs results
cp /path/to/workload_qa_30.jsonl datasets/
wc -l datasets/workload_qa_30.jsonl
find /path/to/any-precision-llm -type d -name 'anyprec-*' -print
```

`find` 的结果中，应选择已打包、能用 `AnyPrecisionForCausalLM.from_quantized` 加载的目录，而不是梯度或 LUT 缓存目录。记录它的完整 WSL 路径。之前量化使用 `--seed_precision 3 --parent_precision 8 --dataset c4 --seq_len 512 --num_examples 100 --random_state 42 --cpu_only`；其中 `--cpu_only` 属于量化流程，推理仍使用 CUDA。

## 配置 A：FP16 基线

新建 `configs/tinyllama_fp16_workload.yaml`：

```yaml
run_name: tinyllama-fp16-workload
output_dir: ../results
repeats: 3
warmup_runs: 2
resource_sample_interval_s: 0.1
workload_ranges:
  - {id: small, min_tokens: 0, max_tokens: 128}
  - {id: medium, min_tokens: 129, max_tokens: 512}
  - {id: large, min_tokens: 513}
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
    extra_options: {dtype: float16}
datasets:
  - id: workload-qa-30
    path: ../datasets/workload_qa_30.jsonl
    task: qa
    metrics: [exact_match, contains_reference, token_f1]
    prompt_templates: ['{prompt}']
```

如果本地已有 FP16 模型快照，可以把 `model` 改成其绝对目录。不要填写 Ollama tag。

## 配置 B：原生 packed w8→w3

新建 `configs/tinyllama_packed_workload.yaml`，**替换 `model` 为实际 packed 目录**：

```yaml
run_name: tinyllama-packed-workload
output_dir: ../results
repeats: 3
warmup_runs: 2
resource_sample_interval_s: 0.1
workload_ranges:
  - {id: small, min_tokens: 0, max_tokens: 128}
  - {id: medium, min_tokens: 129, max_tokens: 512}
  - {id: large, min_tokens: 513}
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
    prompt_templates: ['{prompt}']
```

六个位宽复用同一 packed 模型。这里的顺序是 8→3；如果研究运行顺序的影响，再做一份独立的 3→8 配置并分别保存结果。两份配置的题目、模板、生成参数和 workload 边界需相同。

当前仓库 `src/apbench/metrics.py` 的 `rouge_l` 在预测与参考答案都非空且无共同 token 时会除以零，因此示例暂不启用；若你的本地版本已修复可加回。数据 JSONL 每行应有 `id`、`prompt`、`reference`。

## 运行

```bash
apbench validate configs/tinyllama_fp16_workload.yaml
apbench validate configs/tinyllama_packed_workload.yaml
test -f datasets/workload_qa_30.jsonl

# 每条命令启动独立进程；等待上一条结束再运行下一条
apbench run configs/tinyllama_fp16_workload.yaml
apbench run configs/tinyllama_packed_workload.yaml
```

`validate` 检查配置格式，但不会验证模型文件、数据文件或 GPU 推理。预检时，可把两份配置的 `repeats` 暂改为 1，并在 dataset 下添加 `max_samples: 2`；确认结果后恢复。正式参数下，一份数据 30 题 × 1 模板 × 3 重复，对 FP16 是 90 条 QA 记录，对 packed 的每个位宽是 90 条、合计 540 条。若存在 `__warmup__` 或 `__model_load__` 错误，须检查后再统计。

正式 QA：将数据复制为 `datasets/formal_qa.jsonl`，复制两份 YAML 并修改 `run_name`、dataset 的 `id`、`path`。如需 prompt 敏感性对照，在两份配置中加入**相同的多个模板**；模板数为 3 时记录数乘 3。只使用一个模板不会产生有意义的跨模板比较。

## 输出和解释

每次运行输出到 `results/<run-name>-<UTC时间>/`：

| 文件 | 内容 |
| --- | --- |
| `raw_results.jsonl` | 每题的输出、指标、token、耗时、资源和错误 |
| `summary.csv` | 按模型、精度、数据集、模板汇总 |
| `workload_summary.csv` | 按 small / medium / large 再分组；含样本数、TTFT、TPS、显存 |
| `prompt_sensitivity.csv` | 同题跨模板的输出变化 |
| `baseline_comparison.csv` | **仅在同一次运行含 FP16 和 packed 时**自动计算差值 |
| `run_metadata.json`、`report.md` | 配置元数据和可读报告 |

因为本指南把 FP16 和 packed 分成两个进程，**不会自动得到跨进程的 baseline 对比表**。比较时按两份结果中相同的 `dataset_id`、`case_id`、`template_id`、`repeat` 对齐原始记录，或按对应分组比较两个 `summary.csv` / `workload_summary.csv`；保留运行目录和先后顺序。

- `exact_match` 要求规范化后的完整答案相同。预测为 `Answer: 115`、参考为 `115`，得分为 0；`contains_reference` 会为 1，但可能误判含有正确数字的错误回答。预先指定主要准确率口径并抽样核查。
- `ttft_s` 包含请求到首个输出 token 的时间；即使生成的首个 token 是 EOS，解码后的答案仍可能为空。`tps = (output_tokens - 1) / (wall_time_s - ttft_s)`；少于两个输出 token 时没有 TPS。`tokens_per_second` 是另一种以生成总时间计算的速率，勿混用。
- workload 按**实际输入 token 数**分类：small 0～128，medium 129～512，large 513 起。数据集中的 medium/large 题目名称并不保证分组；检查 `workload_summary.csv` 中的 `sample_count` 和 `mean_input_tokens`。
- `peak_gpu_memory_used_mib` 来自 `nvidia-smi` 在**单次生成期间**对整卡显存的采样峰值，不包含整个模型加载过程的峰值，也不是该精度独占显存。`torch_allocated_bytes`、`torch_reserved_bytes` 属于 PyTorch 分配器口径。

### 显存偏差与模型释放

当前 `runner.py` 在各模型结束后调用 `backend.close()`；现有后端没有实现显式卸载。因此在同进程先跑 FP16、再跑 packed 时，存活引用或 PyTorch 缓存可能影响后续整卡显存。仅添加 `del backend; gc.collect(); torch.cuda.empty_cache()` 不保证释放仍被引用的模型张量，也不会改变 packed 后端**一次加载同一权重服务多个位宽**的设计。独立进程加空闲 GPU 基线更适合比较 FP16 与 packed 的驻留情况；w3～w8 显存相近不能据此断言精度切换失败。要量某一位宽的独占加载成本，需要另设计真正只加载该位宽的模型与相应后端。

## 后续研究检查项

1. 分开报告质量、空答案率、截断比例、TTFT/TPS、运行时显存、packed 文件大小，以及梯度/量化/pack 阶段的时间和资源。
2. 记录基础模型修订号、量化配置、CUDA/PyTorch/Transformers 版本及测试的 GPU 型号；固定数据、模板与生成设置。
3. 比较 8→3 和 3→8 两种顺序；每次启动前记录 `nvidia-smi` 空闲显存，不把不同实验的整卡峰值误当模型固有大小。
4. 将多个等义模板用于正式 QA 后，再评价 `prompt_sensitivity.csv` 和不同模板下的准确率、空回答率。

本指南中的路径均为占位符。实际测量结果以你的 WSL 机器、已打包模型和 `raw_results.jsonl` 为准。

## PMPD 静态精度调度（TinyLlama）

新增 `backend: pmpd`，使用 [SamsungLabs/PMPD](https://github.com/SamsungLabs/PMPD) 的 `PMPDForCausalLM(use_anyprec=True)` 和 `Scheduler.get_scheduler('naive', ...)`。沿用**同一份** TinyLlama seed3/parent8 packed 目录。预填充用 w8，解码最初用 w8，之后按 `high_bit_steps` 降至 w3；不需要下载六份独立模型。此接入目前仅支持静态 `naive` 调度和贪心生成；仓库中的 `kv_cache` 学习式调度需要另外训练分类器，尚未包含在本后端中。

在已经能运行原生 Any-Precision 的 Python 3.11 WSL 虚拟环境里安装 PMPD：

```bash
git clone https://github.com/SamsungLabs/PMPD.git /path/to/PMPD
python -m pip install -e /path/to/PMPD --no-deps
python -c 'from pmpd import PMPDForCausalLM, Scheduler; print("PMPD import OK")'
```

PMPD 上游的 `model.py` 和 `zoo.py` **即使用 Any-Precision 路径也直接导入 AutoGPTQ**。如果最后一条命令报告 `No module named 'auto_gptq'`，可在 PMPD 项目根目录应用本项目的 [可选导入补丁](patches/pmpd_optional_autogptq.patch)，然后重新检查导入：

```bash
cd /path/to/PMPD
git apply --check /path/to/Testing_Demo/patches/pmpd_optional_autogptq.patch
git apply /path/to/Testing_Demo/patches/pmpd_optional_autogptq.patch
python -c 'from pmpd import PMPDForCausalLM, Scheduler; print("PMPD import OK")'
```

该补丁只把 GPTQ 导入设为可选，适用于这里的 `use_anyprec=True`；如果原仓库源码已有变化，先看 `git apply --check` 输出再调整。完整运行仍依赖已安装的 PyTorch、Transformers、Any-Precision CUDA 扩展，以及 PMPD 调度器导入所需的 NumPy。直接安装 PMPD 全部 `requirements.txt` 会引入训练、评测等额外依赖，应先确认与现有环境版本兼容。PMPD 的包装器内部固定调用 `.cuda()`。

编辑 [PMPD 示例配置](configs/tinyllama_pmpd_example.yaml) 的 `model` 为已打包模型的 WSL 绝对路径。示例使用仓库自带的两题数据进行预检：

```bash
cd /path/to/Testing_Demo
apbench validate configs/tinyllama_pmpd_example.yaml
apbench run configs/tinyllama_pmpd_example.yaml
```

查看结果中的 `__model_load__`、`__warmup__` 和普通样本的 `error`。正式对比时，复制配置，把 `path` 改成**与原生 packed / FP16 运行相同**的 QA/workload JSONL，移除 `max_samples`，统一 `repeats`、模板、`max_new_tokens`，给每个 `high_bit_steps`（例如 0、16、32）分别设置不同 `run_name`，每次用独立进程运行。与固定 w8、固定 w3 的 Any-Precision 基线比较时使用相同题目和参数；PMPD 的学习式调度或 FP16 基线应另外记录。

`raw_results.jsonl` 每条 PMPD 记录包括 `schedule`、`prefill_precision`、`high_precision`、`low_precision`、`high_bit_steps`、`token_precision_trace` 和 `token_precision_counts`。`precision_bits: 8` 是这组调度的**标识位宽**，不是所有生成 token 都使用 w8；看轨迹才能确认真正切换。轨迹中首个生成 token 来自 w8 预填充的 logits，后续 token 标注实际产生 logits 的前向计算位宽。`ttft_s` 包含 tokenizer 和预填充，`tps` 使用首 token 之后的输出速度。PMPD 与固定精度复用同一 packed 权重，所以运行时常驻显存未必随每个 token 的位宽变化；重点比较质量、TTFT、TPS、精度轨迹和运行时峰值，并分别运行以避免模型驻留干扰。

这里的生成循环调用 PMPD 的真实模型和静态调度器，但修正上游 `generate()` 多执行一次前向、可能超过 `max_new_tokens`、将“下一次计划精度”计作“已执行精度”的情况。生成口径因此与 PMPD 原脚本的原始 `precision_log` 不完全相同。当前工作区没有 TinyLlama packed 模型或 CUDA GPU，只有配置和模拟推理验证；实际性能和精度仍需要在你的 WSL 上运行。
