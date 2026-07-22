# PK 对比报告模板

> 数据来自 server_compare.json / client_compare.json / bands_compare.csv。
> 纪律：先证可比性；总记分牌标胜者；小样本档打星号；用延迟预算解释端到端结果；一句话结论置顶。

---

# PK 报告：<A> vs <B>（<拓扑/配置>，<并发> 并发 <负载描述>）

一句话结论：**<谁在什么维度赢多少、什么维度输多少、端到端净效果如何、为什么>。**

## 0. 对比前提：两次 run 负载是否一致（可比性检查）

| | <A> | <B> |
|---|---|---|
| 压测时间 | | |
| 会话 / 完成轮次 | | |
| input token 总量 | | |
| output token 总量 | | |
| 缓存命中率 | | |
| 失败请求 | | |

> 若任一行差异 >5%，说明负载不一致，横比结论要降级或放弃。

数据来源：（两个 run 目录 + 客户端 JSON + 本目录脚本清单）

## 1. 总记分牌

| 指标 | <A> | <B> | 胜者 |
|---|---|---|---|
| **端到端总时长** | | | |
| 客户端 output 吞吐（全程平均） | | | |
| **纯净单请求 decode 速度** | | | |
| 每轮 decode 速度中位数（客户端） | | | |
| decode 流式总耗时（req·h） | | | |
| ITL 抢占毛刺占比 | | | |
| 集群 decode 吞吐峰值 | | | |
| **prefill input throughput（中位数/均值）** | | | |
| TTFT mean / median / p99 | | | |
| TTFT 总耗时（req·h） | | | |
| TTFT 占总延迟比例 | | | |
| 开局 queue / pending 峰值 | | | |
| KV full usage 峰值 | | | |

+ 延迟预算图（pk09）：decode 侧省下的 req·h vs TTFT 侧多花的 req·h。

## 2. Decode 对比

图 pk04（速度-并发散点，最干净的 kernel 对比）、pk08、pk01（排空曲线）。

分并发档表（bands_compare.csv），**墙钟占比 <1% 的档打星号并注明不可比**：

| 并发区间 | <A> (tok/s) | <B> (tok/s) | 差异 |
|---|---|---|---|

连锁效果：并发 ≥400/100/50 的维持时间对比、每轮 decode 流式时间中位数、抢占毛刺变化。

## 3. Prefill 对比

图 pk05、pk07。日志侧 input throughput 与客户端侧 TTFT 必须方向一致（交叉验证），并解释幅度差异的原因（TTFT 含调度/缓存匹配等固定开销）。

## 4. 端到端解释：延迟预算算术

每轮 ≈ TTFT + decode 流式时间：
- <A>：≈ 轮数 × (<TTFT_A> + <dec_A>) s
- <B>：≈ 轮数 × (<TTFT_B> + <dec_B>) s

decode 项省 <X> s/轮，TTFT 项多花 <Y> s/轮，净变化 <Z> s/轮 —— 与实测总时长差异 <P>% 是否吻合。
再用 pk06 墙钟分布印证瓶颈转移。

## 5. 建议

1. （取长补短的工程方案）
2. （下一步 microbenchmark 定位）
3. （对在线服务 vs benchmark 的不同含义）
4. （benchmark 方法论改进）

## 附：文件清单
