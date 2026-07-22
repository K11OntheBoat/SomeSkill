---
name: sglang-log-profile
description: 对 SGLang（或类似 LLM 推理服务）压测产生的服务端日志 + 客户端 benchmark JSON 做全程性能分析，产出时间序列 CSV、图表和 REPORT.md；支持两个服务/backend 的 PK 对比报告。当用户说"分析这次压测日志"、"给两个 backend 做 PK"、"产出压测报告"时使用。
---

# SGLang 压测日志分析与 PK 报告

这个 skill 把一次完整的压测分析固化为可复现的流水线。**核心思想：先把海量日志压缩成 60s 分桶的时间序列 CSV（中间产物、可交叉验证），再基于 CSV 画图和写结论，绝不直接从原始日志跳到结论。**

## 输入（开始前先向用户确认/自己寻找）

1. **服务端日志**：SGLang worker 日志（含 `Prefill batch` / `Decode batch` 行），可能分散在多个节点文件。
2. **客户端 benchmark JSON**（可选但强烈建议）：FastDeploy/bench 工具输出，含 `ttfts / input_lens / output_lens / itls / duration / completed` 等字段。用于交叉验证服务端结论。
3. **运行元信息**：并发数上限、DP/TP/EP 拓扑、`decode_log_interval`（决定吞吐是精确还是估算，见下）、模型和负载描述（如"768 并发 256K SWE 多轮"）。

## 输出目录约定

每个 run 一个目录（如 `<结果根目录>/<并发>_<标签>/`），PK 再单开一个目录：

```
<run_dir>/
  data/batches_all.log      # 预过滤的 batch 行（可复现的分析输入）
  timeseries_60s.csv/.txt   # 60s 分桶时间序列（一切分析的基础）
  per_dp_stats.csv          # 每 DP 生命周期统计
  dp_heatmap.csv            # DP × 时间 热力矩阵
  summary.json              # 全局汇总
  client_summary.json       # 客户端侧汇总
  plots/*.png               # 编号图表
  REPORT.md                 # 单 run 报告（模板见 references/）
<pk_dir>/
  server_compare.json / client_compare.json / bands_compare.csv
  plots/pk*.png
  REPORT.md                 # PK 报告（模板见 references/）
```

## 流水线（依次执行）

### Step 1 — 提取 batch 行

```bash
scripts/extract_batches.sh '<原始日志glob>' <run_dir>/data/batches_all.log
```

原始日志可能有几 GB，必须先 grep 过滤，后续所有分析只读这个过滤文件。

### Step 2 — 服务端全程分析

```bash
python3 scripts/analyze_server.py --input <run_dir>/data/batches_all.log \
    --outdir <run_dir> --concurrency-cap <并发上限>
```

产出 timeseries/per_dp/heatmap/summary + plots/01–08。

**格式适配**：脚本顶部的 `P_RE`/`D_RE` 正则对应当前 SGLang 日志格式。如果目标服务日志字段不同（不同版本/不同引擎），先 `head` 几行样本，改正则和字段映射，其余管线不用动。跑完检查 `bad=0`（解析失败行数），bad 比例高说明正则不匹配，不能带病出数。

**吞吐口径（关键，写进报告）**：若 `decode_log_interval=1`，每条 Decode 行 = 一个 decode step，恰好生成 `#running-req` 个 token，故 `sum(running)/60s` 是**精确**集群 tok/s。若 interval>1，只能用"平均瞬时 gen throughput × DP 数"**估算**，报告里必须注明口径。

### Step 3 — 判定活跃窗口，画放大图

开环 benchmark 跑完后常有健康探测长尾（特征：`#new-seq` 恒定小值、`cached-token=0`、每 DP 均匀一条）。看 `timeseries_60s.txt` 找到真实负载结束点（最后一段 `p_cached>0` 的时刻），然后：

```bash
python3 scripts/plot_active.py --dir <run_dir> --end-s <活跃窗口秒数> \
    --concurrency-cap <并发上限>
```

产出 plots/09–12（活跃窗口放大：并发排空曲线、精确吞吐、每请求速度、并发-时长 CDF）。

### Step 4 — 客户端分析（有 JSON 时）

```bash
python3 scripts/analyze_client.py --json <benchmark结果.json> --outdir <run_dir>
```

产出 client_summary.json + plots/13–15（TTFT 分布、TTFT vs 输入长度、每请求延迟构成 TTFT/decode 堆叠）。

### Step 5 — 交叉验证（写报告前必做）

服务端和客户端两条独立数据链必须互相印证，否则先查数据再下结论：
- 客户端 `total_output_tokens` ≈ 服务端精确 decode token 计数（误差 <5%）；
- 客户端 TTFT 变化方向 ≈ 服务端 prefill input throughput 变化方向；
- 客户端 duration ≈ 服务端活跃窗口长度。

### Step 6 — 单 run 报告

按 `references/report_single_template.md` 写 `<run_dir>/REPORT.md`。必答问题清单：

1. 全局画像表（时长、轮次、token 量、缓存命中率、并发峰值、队列峰值、KV 峰值、异常计数）。
2. 并发时间线是什么形态（满载稳态 vs 单调排空）？分阶段表。
3. **并发-时长分档表**（700+/400-700/200-400/100-200/50-100/10-50/0-10）：墙钟占比、token 占比、每请求速度。这是判断"长尾是否主导"的核心证据。
4. 瓶颈定位：decode/prefill/KV/队列/路由，每个结论都要有图或数支撑。
5. 负载均衡：per_dp_stats + 热力图。
6. 结论与建议（区分"benchmark 方法论问题"和"服务真实瓶颈"）。

### Step 7 — PK 对比（两个 run 都完成 Step 1–6 后）

```bash
python3 scripts/compare_server.py --outdir <pk_dir> \
    --run <名A>=<run_dir_A>=<活跃秒A> --run <名B>=<run_dir_B>=<活跃秒B>
python3 scripts/compare_client.py --outdir <pk_dir> \
    --run <名A>=<jsonA> --run <名B>=<jsonB>
```

按 `references/report_pk_template.md` 写 PK 报告。PK 报告纪律：

- 先证明**可比性**（两次 run 的会话数/轮次/token 量/命中率几乎一致），不一致就不能直接横比。
- 总记分牌表：每行一个指标、标注胜者。
- 分并发档对比每请求 decode 速度（`bands_compare.csv`），**样本太少的档要打星号剔除**（如某档墙钟占比 <1%）。
- 用**延迟预算**解释端到端结果：每轮 ≈ TTFT + decode 流式时间，算出每轮各省/多花几秒，验证与实测总时长差异吻合——这是把"A 快 91% 但总时长打平"这类反直觉结果讲清楚的标准手法。
- 一句话结论放最顶部。

## 常见坑（历史教训，务必检查）

- **只看前 1 小时会得出错误的"稳态"结论**：必须分析全程再判断形态。
- **瞬时 gen throughput 有毛刺**（受 prefill 插入影响），不要用它评估持续吞吐，用精确 token 计数。
- **开环排空型 benchmark 不适合评估满载稳态吞吐**：>N 并发的维持时间可能只有几十分钟，报告里要写明；要测稳态需 closed-loop 补位。
- **DP 失衡结论要用全程数据**，早期窗口的瞬时倾斜不算数。
- 日志尾部的 gloo "Connection reset" 通常是服务停止时正常断连，不是故障。
- 客户端 JSON 中的自定义字段（`s_decode_clean`、`n_itls_preempt`）不一定存在，脚本已做兼容，缺失时报告里跳过对应行。
