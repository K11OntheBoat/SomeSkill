# 样例结论（DeepSeek-V4-Flash on SGLang，TP/DP/EP-0 rank）

这是一份真实 trace 分析产出的关键数字，用来对照"报告长什么样、结论怎么下"。数字本身依 run 而异，别照抄，学的是**读法**。

## 三阶段概览（来自 01/03）

| Phase | Batches/Steps | CPU wall (avg) | GPU wall (avg) | kernel_sum (avg) | kernel util |
|-------|---------------|----------------|----------------|------------------|-------------|
| Extend (Prefill) | 6 | 452.72ms | 449.69ms | 148.94ms | 33.1% |
| Decode-Eager | 25 | 459.93ms | 456.55ms | 86.40ms | 18.9% |
| Decode-CUDAGraph | 34 | 5.99ms | 16.41ms | 16.61ms | 101.2% |

读法：
- Extend/Decode-Eager 的 kernel util 只有 ~20-33% → GPU 大量 idle，瓶颈不在算力。
- Decode-Eager 单步 456ms vs CUDAGraph 单步 16ms → **eager 慢 ~77x**，每个漏进 eager 的 decode batch 都是巨大损失。
- CUDAGraph gap 为负、util >100% → CPU 提前发完、GPU 被多 stream 喂满，健康。

## GPU kernel 占比（来自 02）

- Extend：通信 42.9% / Attention 20% / GEMM 18.8%。prefill 通信和计算比较均衡。
- Decode-Eager：**通信 83.1%**，其中 `notify_dispatch` 单个 kernel 就 1.029s（占该阶段一半以上）→ decode 被 DeepEP 跨节点通信主导。
- Decode-CUDAGraph：GEMM 37% / 量化 19.6% / 通信仅 13.6% → 命中 graph 后回到计算密集，通信被压下去。

## 系统瓶颈（来自 04）

- 全局 GPU 利用率 23.6%，**有效计算利用率仅 8.9%**（compute-only/wall）。
- DeepEP 通信 2.254s 分解：notify_dispatch 47.5% / combine 21.3% / dispatch 19.9% / cached_notify 11.3%。
- CUDA Graph 覆盖 34/59 decode batch（57.6%），加速比 77x。
- 调度开销：get_next_batch 469ms + process_batch_result 659ms，合计 >1s。

## 优化优先级（结论模板）

1. **降 DeepEP 通信**：notify_dispatch 是等远端就绪的同步，方向是通信-计算 overlap（shared expert 与 dispatch 并行），不是调 kernel。
2. **提 CUDA Graph 覆盖率**：查为什么部分 batch size 没命中 capture。
3. **减 CPU 调度开销**：graph 模式下 CPU 调度可能反成瓶颈。

## per-layer（来自 05）

- 大多数层耗时稳定（std 几百 us），Layer 0 方差极大（avg 23ms / max 96ms / std 26ms）→ 首层常含额外同步/warmup，单独看待。
- 奇偶层交替：偶数层（C4）~10.7ms，奇数层（MLA）~9.3ms，结构不同导致规律性差异。
- MoE 内部：DeepEPMoE 73.5% / shared MLP 17.7% / TopK 6.7% / Gate 2.1% → MoE 时间几乎全在专家计算+通信。
