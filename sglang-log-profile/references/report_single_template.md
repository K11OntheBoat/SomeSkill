# 单 run 报告模板

> 结构参考。数据全部来自 summary.json / client_summary.json / timeseries_60s.csv / bands 分析。
> 每个结论必须有图或数字支撑；写清吞吐口径（精确 vs 估算）。

---

# <服务/配置> — <并发> 并发 <负载描述> 压测全程日志分析

分析对象：`<原始日志路径>`（<文件数/大小>，<N> 条 batch 日志：prefill <N> 条 + decode <N> 条，解析失败 <N> 条）。

生成工具（本目录）：
- `analyze_server.py` → `timeseries_60s.csv/.txt`、`per_dp_stats.csv`、`dp_heatmap.csv`、`summary.json`、`plots/01~08`
- `plot_active.py` → 活跃窗口放大图 `plots/09~12`
- `analyze_client.py` → `client_summary.json`、`plots/13~15`
- `data/batches_all.log`：预过滤 batch 行（分析输入，可复现）

## 0. 全局画像

| 指标 | 数值 |
|---|---|
| 日志覆盖时间 | <t_start> → <t_end>（<X> h） |
| **真实压测窗口** | <开始> → <结束>（<X> h），之后为健康探测/空载 |
| 完成的 prefill 轮次（#new-seq） | <N>（<会话数> 会话 ≈ 平均 <X> 轮/会话） |
| prefill token | new = <X> M，cached = <X> M，**缓存命中率 <X>%** |
| decode 生成 token（口径：精确/估算） | <X> M（≈ <X> tok/轮） |
| decode 并发峰值 | <X>（上限 <cap>；每 DP 上限 <X>） |
| 服务端排队峰值 | prefill queue <X> / decode queue <X>，出现时段 <...> |
| KV 利用率 | full usage 峰值 <X>，平均 <X> |
| 异常 | <retract/OOM/断连 计数及定性> |

## 1. 全程时间线：并发形态判定

<核心判断：满载稳态 or 单调排空？> + 活跃窗口并发图（plots/09）

| 阶段 | 时间 | 活跃 decode 并发 | 特征 |
|---|---|---|---|
| ① 灌入 | ... | ... | ... |
| ② ... | | | |

按小时汇总表（平均并发 / 集群 tok/s / 生成 token / 新轮次）。

## 2. 并发-时长分布：长尾是否主导

| 并发区间 | 墙钟占比 | decode token 占比 | 每请求 decode 速度 |
|---|---|---|---|
| 700+ | | | |
| 400–700 | | | |
| 200–400 | | | |
| 100–200 | | | |
| 50–100 | | | |
| 10–50 | | | |
| 0–10 | | | |

+ 并发时长 CDF 图（plots/12）。回答：端到端时长由吞吐决定还是由长尾决定？

## 3. Decode 侧：吞吐与每请求速度

图 plots/10、11、08。回答：
- 集群吞吐峰值 / 满载时每请求速度；
- 每请求速度随并发/上下文长度如何变化；
- decode 是否为瓶颈。

## 4. Prefill / KV / 队列侧

图 plots/03、05。回答：缓存命中率、KV 是否打满、队列是否非零、prefill 是否制约。

## 5. 负载均衡（DP 间）

per_dp_stats.csv + plots/07 热力图。用全程数据下结论，注明早期瞬时倾斜不算数。

## 6. 客户端交叉验证

client_summary.json + plots/13–15。核对 output token 总量、TTFT/decode 延迟拆分（TTFT 占比 <X>%）。

## 7. 结论与建议

编号列出。区分：(a) 服务真实瓶颈及优化方向；(b) benchmark 方法论问题（如开环排空不适合测满载稳态）。

## 附：文件清单

（列出本目录所有产物及一句话说明）
