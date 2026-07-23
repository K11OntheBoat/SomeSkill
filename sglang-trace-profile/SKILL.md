---
name: sglang-trace-profile
description: 分析 SGLang（DeepSeek-V4/类似 MoE 模型）用 PyTorch Profiler 抓取的 Chrome Trace JSON（单 rank .trace.json），按 Extend/Decode-Eager/Decode-CUDAGraph 三个阶段拆解 CPU 调用栈、GPU kernel、CPU-GPU gap、系统瓶颈（DeepEP 通信、CUDA Graph 覆盖率、调度开销）等，产出 5 份 .md 报告。当用户说"分析这个 profile trace"、"看看 kernel 耗时/瓶颈在哪"、"拆解一下 decode 的调用栈"、"这次 profiling 结果" 时使用。区别于 sglang-log-profile（那个是压测日志时间序列分析，这个是单次 profiler trace 的算子级拆解）。
---

# SGLang Profiler Trace 算子级分析

把一次 PyTorch Profiler trace（Chrome Trace Format JSON）的分析固化为可复现流水线。**核心思想：一份原始 trace 可能有 1GB+，先按 category（`python_function` / `kernel` / `gpu_user_annotation` / `cuda_runtime` / `gpu_memcpy`）分组并识别主线程/主 GPU stream，再按 batch/step 时间窗口做区间聚合，最后按三个执行阶段分别成文——绝不把所有 rank、所有阶段混在一起看。**

这个 skill 与 [[sglang-log-profile]] 互补：那个从服务端日志看全程宏观吞吐/并发，这个从单次 profiler trace 看微观算子/调用栈/瓶颈。两者结论可互相印证（如 log 侧看到 decode 慢 → trace 侧定位到 DeepEP 通信占 83%）。

## 输入（开始前先确认）

1. **一份 Chrome Trace JSON**：SGLang 用 `torch.profiler` 抓取，文件名形如 `<ts>-TP-0-DP-0-EP-0.trace.json`。**只分析一个 rank（通常 TP-0/DP-0/EP-0），不要解压全部**——每个 rank 100MB+（gz），解压后 1GB+，一个 rank 足够定位算子瓶颈。其余 rank 的 `.gz` 保留即可。
2. **模型/拓扑信息**：模型名（如 DeepSeek-V4-Flash）、TP/DP/EP 拓扑、是否开 DeepEP、CUDA Graph 的 capture batch sizes。用于解读 per-layer 结构与通信算子。

trace 里关键的 event 结构（脚本依赖，换版本前先核对）：
- `ph == 'X'` 的区间事件，按 `cat` 分类；
- `python_function`：Python 调用栈（含 `scheduler.py` / `model_runner` / `nn.Module: XxxLayer_N`），主线程 = 出现 `scheduler.py`/`model_runner` 最多的 tid；
- `kernel`：GPU kernel，按 name 分类为 Communication/GEMM/Attention/... ；
- `gpu_user_annotation`：`step[...EXTEND...]` / `step[...DECODE...]` 标注，主 GPU stream = 同时含 extend+decode 且 step 最多的 tid。

## 产出（5 份报告，写到 trace 同目录或 `--output-dir`）

| 报告 | 内容 | 回答什么问题 |
|------|------|-------------|
| `01_cpu_stack.md` | 三阶段 CPU 调用栈树（最长 batch 取样）+ per-layer 耗时表 + MQALayer/MoE 子组件拆解 + 代表层（奇偶层 attention 结构不同）内部详解 | CPU 侧时间花在哪个 Python 调用/哪一层/哪个子模块 |
| `02_gpu_stack.md` | 三阶段 GPU kernel Top-10 + 按类别（通信/GEMM/Attention/量化/Norm/...）占比 + 通信/计算/Attention 核心占比 | GPU 侧哪些 kernel 最贵、通信 vs 计算比例 |
| `03_cpu_gpu_gap.md` | 每阶段 CPU wall vs GPU wall vs kernel_sum，算 gap 和 kernel utilization，判定瓶颈在 CPU launch 还是 GPU/通信 | GPU 是否被喂饱、瓶颈在 CPU 调度还是 GPU 执行 |
| `04_bottleneck_analysis.md` | 全局 GPU 利用率、DeepEP 通信分解（notify_dispatch/dispatch/combine/cached_notify）、CUDA Graph 覆盖率与加速比、调度开销、优化建议 | 系统级最大瓶颈是什么、优先做什么优化 |
| `05_extra_analysis.md` | per-layer 耗时方差、CUDA Runtime launch overhead（cudaLaunchKernel 等）、GPU memcpy、MoE 内部组件（DeepEPMoE/shared MLP/TopK/Gate）汇总 | 层间是否均衡、launch 开销、访存、MoE 内部构成 |

## 流水线

### Step 1 — 准备单个 rank 的 trace

trace 通常是 `.trace.json.gz`。只挑一个 rank 解压（若已有解压好的 `.trace.json` 直接用）：

```bash
# 仅解压 TP-0（保留 .gz 原文件）
gunzip -k <dir>/<ts>-TP-0-DP-0-EP-0.trace.json.gz   # 若已是 .json 跳过
ls -lh <dir>/*.trace.json
```

### Step 2 — 跑分析器

```bash
python3 scripts/analyze_trace.py <dir>/<ts>-TP-0-DP-0-EP-0.trace.json \
    [--output-dir <报告目录，默认 trace 同目录>]
```

脚本会打印：主线程 tid、主 GPU stream tid、decode 阶段 eager/graph 的**自动阈值**（用相邻耗时最大 gap 切分，>5x 才认为存在明显分界，否则退回 100ms 固定阈值）、以及 CPU batches / GPU steps 的三阶段计数。**跑完先核对这几行**：若 Extend/Decode 计数为 0 或主线程识别失败，说明该 trace 的命名/字段与脚本假设不符，需按下节适配，不能带病出报告。

只用标准库（json/os/re/collections/bisect/dataclasses），无需额外依赖。1GB trace 的 `json.load` 需要较大内存（~数 GB）和几十秒到几分钟，属正常。

### Step 3 — 读报告、下结论

按上表逐份读。典型结论链（来自 DeepSeek-V4-Flash 实测样例，见 `references/`）：
- Decode-Eager 阶段 GPU kernel 83% 是 DeepEP 通信，其中 `notify_dispatch`（等远端就绪）占通信近半 → **通信是 decode 最大瓶颈**；
- CUDA Graph 相对 Eager 有数十倍加速（样例 77x），但覆盖率不足（样例 34/59）→ **提升 CUDA Graph 命中率是第二优先**；
- 整体 GPU 利用率低（样例 23.6%，有效计算利用率 8.9%）→ 大量 GPU idle，等 CPU launch / 等通信。

## 阶段划分逻辑（理解报告的前提）

脚本把执行切成三个阶段，务必在报告里讲清区别：

- **Extend (Prefill)**：CPU 侧 `run_batch` 内出现 `_execute_extend`；GPU 侧 step 名含 `EXTEND`。处理输入 prompt，batch 大、单步耗时长。
- **Decode-Eager**：decode 但**没走 CUDA Graph**（Python 一层层 eager 执行）。GPU step 耗时落在阈值**高**侧。每个 eager decode step 都很贵（样例 ~456ms/step），是明确的性能损失。
- **Decode-CUDAGraph**：decode 且命中 CUDA Graph replay，Python 层被跳过（所以 `01` 报告里这个阶段没有 per-layer 事件）。GPU step 耗时落在阈值**低**侧，快数十倍。

## kernel 分类规则（`classify_kernel`，改前先看样例名）

按 name 关键字归类，顺序敏感（先匹配先归类）：
- **Communication**：`nccl`/`deep_ep`/`allreduce`/`allgather`/`broadcast`/`reduce_scatter`/`send`/`recv`/`internode`；
- **Attention**：`flash`/`attention`；单独的 `mla`（非 gemm）、`mqa_logits`、`mla_combine`；
- **GEMM**：`gemm`/`nvjet`/`cutlass`；
- 其余：Quantization/Norm/RoPE/Activation/TileLang/MHC/Elementwise/TopK-Softmax/Other。

换模型/换 kernel 库（如非 DeepEP、非 deep_gemm、非 tilelang）时，**先看 `02` 报告里 Other 类别占比**：Other 偏高说明分类规则没覆盖该模型的算子，需在 `COMM_KEYWORDS` 和 `classify_kernel` 里补关键字，否则"通信/计算占比"会失真。

## 适配到不同 SGLang 版本 / 不同模型（重要）

脚本对 DeepSeek-V4-Flash + 特定 SGLang 版本的命名有硬编码假设，换环境时按需改（都在 `scripts/analyze_trace.py` 顶部/各 extract 函数里，位置已注释）：

- **阶段判定关键字**：`_execute_extend`、`decode_cuda_graph_runner`（`classify_phases`）；step 标注 `step[...EXTEND/DECODE...]`。
- **主线程识别**：靠 `scheduler.py` / `model_runner`（`preprocess_events`）。
- **层/子模块名**：`nn.Module: DeepseekV4DecoderLayer_N`、`MQALayer_N`、`DeepseekV2MoE_N`、`MoEGate/DeepseekV2MLP/TopK/DeepEPMoE/HashTopK`、`C4Indexer`、`mhc_fused_post_pre`（`extract_layer_components` / `generate_01`）。换模型时改成对应模块名。
- **奇偶层结构**：DeepSeek-V4 奇数层 MLA、偶数层 C4，脚本对 Layer 3/4 分别出 attention 内部详解。别的模型没有这个交替就删掉这段或改层号。
- **decode 阈值**：自动检测（相邻耗时最大 gap，>5x 生效），一般不用动；若 eager/graph 耗时差不明显，会退回固定 100ms，此时按实际手动改 `decode_threshold`。

改完重跑，对照 Step 2 的打印计数确认解析正常。

## 常见坑（历史教训）

- **不要解压/加载所有 rank**：一个 rank 就够定位算子瓶颈，全部解压会吃满磁盘和内存。
- **取样 batch 会掐头去尾**：`01` 报告选"最长 batch"作调用栈样例时跳过首尾 batch——首个 batch 常有 Dynamo/Inductor 首次编译、末个有 teardown，会让 `_execute_extend` 的子调用错位成兄弟节点。别用首尾 batch 当代表。
- **CUDA Graph 阶段没有 per-layer Python 事件**是正常的（replay 跳过 Python），不是数据缺失。
- **gap 为负**（CPU wall < GPU wall，见 CUDAGraph 阶段）表示 CPU 提前把 Python 调用发完、GPU 还在排队执行，属正常，不是错误。
- **kernel utilization >100%**（如 CUDAGraph 101%）是因为多 stream 并行 kernel 时间和 > wall，说明 GPU 被喂得很满，是好现象。
- **`notify_dispatch` 高 ≠ dispatch 本身慢**：它是等远端 EP rank 就绪的同步等待，反映的是跨节点负载不均/通信-计算未 overlap，优化方向是 overlap 而非调 dispatch kernel。
- 分类里 **Other 占比高**要警惕分类规则漏了算子（见上节）。
