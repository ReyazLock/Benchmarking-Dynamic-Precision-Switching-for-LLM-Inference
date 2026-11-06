# AnyPrecision Bench

这是一个面向研究实验的 Python 测试框架，用于比较 FP16 与 INT5–INT8 LLM 在质量、资源、速度和 Prompt 稳定性上的差异。它同时支持：

- **Ollama 后端**：测试已经导入 Ollama 的独立量化模型；
- **Any-Precision 原生后端**：从同一个 bitplane packed model 在 5/6/7/8 bit 间切换，直接验证论文的核心机制；
- **Hugging Face 后端**：作为本地 FP16/BF16 基线；
- **量化阶段 profiler**：分别记录 gradients、quantize、pack 三个阶段的时间与资源。

## 必须先明确的兼容性边界

Ollama 不能替代 Any-Precision 的原生 serving engine。Any-Precision 使用自定义 CUDA kernel 和 bitplane 权重表示，在同一个 packed artifact 上动态选择精度；Ollama 通常加载独立的 GGUF 量化文件。因此：

1. 使用 Ollama 可以回答“不同已部署量化模型的实际速度、显存和质量怎样”；
2. 使用原生后端才能回答“同一 Any-Precision 模型能否以低额外存储成本支持 INT5–INT8 动态切换”；
3. Ollama 配置中的 `precision_bits` 是实验标签。框架会把 `/api/ps` 返回的实际 `quantization_level` 写入结果，防止把文件名误当成真实位宽；
4. 常见 GGUF 量化格式并不保证恰好存在论文意义上的每一个整数位宽，特别是 INT7。不要把 `Q5_K_M` 与 Any-Precision 的 5-bit 算法视为完全相同的量化方法。

## 安装

需要 Python 3.11+。

```bash
cd anyprecision-bench
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

仅使用 Ollama 时，上述安装已足够。使用原生 Any-Precision/Hugging Face 后端时：

```bash
pip install -e '.[native]'
cd /path/to/any-precision-llm/any_precision/modules/kernels
pip install .
```

原生 Any-Precision 仍需满足上游项目要求：CUDA Toolkit 12+、gcc 9+，并先生成 packed model。

## 快速开始：Ollama

1. 准备来自同一个 base checkpoint 的 FP16、INT5、INT6、INT7、INT8 模型文件，并分别导入 Ollama。实际可用的 GGUF 类型取决于转换/量化工具；没有真正的 7-bit artifact 时不要用其他格式冒充。
2. 修改 `configs/ollama_example.yaml` 中五个 `model` tag。
3. 检查配置并执行：

```bash
apbench validate configs/ollama_example.yaml
apbench run configs/ollama_example.yaml
```

框架调用 Ollama 的 `POST /api/generate`，记录服务端报告的 token 数、load/prompt-eval/generation 时间；随后通过 `GET /api/ps` 记录驻留模型大小、VRAM 大小和真实量化格式。每个模型完成后会请求卸载，减少前一个模型对后一个模型显存采样的污染。

## 快速开始：真正的 Any-Precision INT5–INT8

先在上游项目中量化：

```bash
python quantize.py /models/base-llama \
  --seed_precision 5 \
  --parent_precision 8 \
  --random_state 42
```

将 `configs/anyprecision_example.yaml` 中的 FP16 checkpoint 和 packed model 路径改为本机路径，然后运行：

```bash
apbench validate configs/anyprecision_example.yaml
apbench run configs/anyprecision_example.yaml
```

一条 `anyprecision` model 配置中的 `precisions: [5, 6, 7, 8]` 会复用同一个已加载模型，并在每次 `generate(..., precision=n)` 或 forward 时选择位宽。这才是对论文动态精度机制的直接测试。

## 测量量化阶段，而不是把它误称为训练阶段

Any-Precision 的 README 描述的是训练后量化流程：梯度敏感度计算、incremental upscaling/quantize、pack；它没有在这里重新训练 base LLM。可用下面的命令分别测量三个阶段：

```bash
apbench profile-quantization \
  --repo /path/to/any-precision-llm \
  --model /models/base-llama \
  --seed-precision 5 \
  --parent-precision 8 \
  --output results/quantization_profile.json
```

如果研究问题确实是“训练前后输出如何变化”，应该把 base checkpoint 与 fine-tuned checkpoints 各自配置成一个 `huggingface` model，并在完全相同的数据、tokenizer 与 generation 参数下比较。训练阶段本身主要产生 logits/loss/gradients，不是与推理文本一一对应的另一种“输出”。

## 数据格式

QA/指令数据集采用 JSONL，每行至少包含：

```json
{"id":"q1","prompt":"What is 2 + 2?","reference":"4","metadata":{"category":"math"}}
```

语言模型/perplexity 数据集：

```json
{"id":"lm1","text":"Text whose next-token perplexity will be evaluated."}
```

Ollama 的 generate API 不能提供输入序列每个真实 token 的完整 logits，所以框架不会用生成 token 的 logprob 冒充标准 perplexity。`language_modeling` 任务只在 Any-Precision 和 Hugging Face 本地后端计算。

## 输出

每次运行生成一个带 UTC 时间戳的目录，其中包含：

| 文件 | 用途 |
| --- | --- |
| `raw_results.jsonl` | 每个 model × precision × case × template × repeat 的原始输出与测量值 |
| `summary.csv` | 平均质量、perplexity、P50/P95 延迟、吞吐、内存/显存等聚合结果 |
| `baseline_comparison.csv` | 各量化方案相对 `baseline: true` 模型的绝对变化和百分比变化 |
| `prompt_sensitivity.csv` | 同一题在不同 Prompt 模板下的输出分歧率与各质量指标跨度 |
| `run_metadata.json` | 完整配置、Python/OS 环境与生成参数 |
| `report.md` | 可快速阅读的实验表格 |

主要资源字段：

- `model_size_bytes` / `model_vram_bytes`：Ollama 对当前模型报告的驻留大小，最适合跨 Ollama variant 比较；
- `model_artifact_size_bytes`：原生 Any-Precision packed artifact 的磁盘大小；
- `peak_gpu_memory_used_mib`：`nvidia-smi` 的整机 GPU 峰值；
- `peak_ollama_rss_bytes`：Ollama 进程 RSS 峰值；
- `wall_time_s`、`tokens_per_second`：端到端延迟与 decode 吞吐；
- `exact_match`、`contains_reference`、`token_f1`、`rouge_l`、`perplexity`：质量指标。

## 对应你的三个研究问题

### 1. 是否真的节省资源，精度是否下降？

把同一个 base checkpoint 的 FP16 设为 `baseline: true`。优先查看 `baseline_comparison.csv`：

- 内存节省率：`mean_model_vram_bytes_change_pct`（负数表示减少）；
- 延迟变化：`p50_wall_time_s_change_pct`；
- 吞吐变化：`mean_tokens_per_second_change_pct`；
- QA 质量变化：`mean_exact_match_delta`、`mean_token_f1_delta`；
- 语言建模退化：`mean_perplexity_change_pct`（正数通常表示变差）。

Any-Precision 的“低成本部署多个尺寸”主张还需要做一个**组合存储实验**：比较一个 INT5–INT8 packed artifact 的磁盘大小，与四个独立 5/6/7/8-bit artifact 的总大小。单独比较 INT5 和 FP16 不能验证 overlay 带来的多精度部署收益。

```bash
apbench compare-storage \
  --packed /models/anyprec-int5-int8 \
  --independent /models/int5 /models/int6 /models/int7 /models/int8 \
  --output results/storage_comparison.json
```

### 2. Prompt 灵敏度是否影响结果？

在 dataset 的 `prompt_templates` 中放入语义相同、措辞不同的模板。查看 `prompt_sensitivity.csv`：

- `unique_output_ratio` 越高，表示模板变化更容易改变输出；
- `exact_match_range`/`token_f1_range` 越大，表示模板变化确实改变任务质量，而不只是措辞；
- 应分别做 temperature=0 的确定性实验与 temperature>0 的采样实验。后者需要多个 seed，框架会令每个 repeat 使用 `seed + repeat`。

### 3. 量化阶段与 inference 阶段如何变化？

`profile-quantization` 回答量化管线的时间和资源代价；`apbench run` 回答产物在 inference 时的文本、质量与资源变化。若需要研究校准数据的影响，可以改变量化命令的 `dataset`、`num_examples` 与 `random_state`，生成多个 packed artifact，再把它们作为不同 model 加入同一 benchmark。

## 推荐的正式实验规程

1. 所有量化模型必须来自同一 base checkpoint，并记录 commit、模型 digest、tokenizer 与 quantizer 版本。
2. 使用至少一个标准任务集；示例 `tiny_qa.jsonl` 只用于 smoke test，不能支撑论文结论。
3. 固定 context length、最大输出 token、batch size、seed 和 Prompt；先 warm-up，再至少重复 5–10 次。
4. 资源实验时保持机器空闲，并分开做 cold-load（包含加载）与 warm inference（排除加载）实验。
5. 同时报告绝对值、相对 FP16 变化和置信区间；不要只报告“最多节省”。
6. 为准确率设定预先阈值，例如“相对 FP16 的任务分数下降不超过 1%”，再判断某个位宽是否可接受。
7. 对 INT5–INT8 使用相同样本顺序；若硬件支持，随机化 model 运行顺序以减少温度/系统状态偏差。

## 当前限制与下一步

- 当前内置指标适合短答案和基础文本重合；代码生成、摘要、对话、安全性需要接入 HumanEval、ROUGE/BERTScore、MT-Bench 或领域数据集。
- `nvidia-smi` 是整卡采样，多租户环境会产生噪声；Ollama 的 `size_vram` 更适合判断单模型驻留大小。
- 当前 runner 串行执行，避免显存竞争。并发 serving、吞吐随 batch/concurrency 的扩展应作为独立实验。
- 原生 backend 按上游 README 的公开 API 编写；若你固定到某个 commit，应在首次运行后用一个小样本验证 `forward(..., precision=)` 与 `generate(..., precision=)` 的签名。

## 测试

```bash
pytest
```

项目包含配置解析、文本指标和 Ollama API 映射的单元测试；不需要下载模型即可运行。
