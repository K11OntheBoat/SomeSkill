#!/usr/bin/env python3
"""Full-run analysis of SGLang worker batch logs.

Input : pre-filtered batch lines (see extract_batches.sh)
Output: (all in --outdir)
  - timeseries_60s.csv / timeseries_60s.txt : aggregate metrics per 60s bucket
  - per_dp_stats.csv                        : per-DP-rank lifetime stats
  - dp_heatmap.csv                          : per-DP avg decode running-req per 10min bucket
  - summary.json                            : run-level summary
  - plots/01..08_*.png                      : full-run time-series charts

NOTE on throughput: with decode_log_interval=1 every Decode line is one decode
step generating exactly #running-req tokens, so sum(running)/60s is the EXACT
cluster decode tok/s. With a larger interval it is only an estimate.

If the target service's log format differs, adapt TS_RE / P_RE / D_RE below
and check the printed `bad=` counter is ~0 after running.
"""
import argparse, re, os, json, datetime, csv
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, help="pre-filtered batch log file")
ap.add_argument("--outdir", required=True)
ap.add_argument("--concurrency-cap", type=int, default=None,
                help="benchmark concurrency cap, drawn as reference line")
ap.add_argument("--bucket", type=int, default=60, help="main bucket width (s)")
ap.add_argument("--heat-bucket", type=int, default=600, help="heatmap bucket width (s)")
args = ap.parse_args()

BASE = os.path.abspath(args.outdir)
SRC = args.input
PLOTS = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)
BW = args.bucket
HEAT_BW = args.heat_bucket

# ---- log format regexes (ADAPT HERE for other services/versions) ----
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) DP(\d+) ")
P_RE = re.compile(
    r"#new-seq:\s*(\d+), #new-token:\s*(\d+), #cached-token:\s*(\d+), "
    r"full token usage:\s*([\d.]+), swa token usage:\s*([\d.]+), "
    r"#running-req:\s*(\d+), #queue-req:\s*(\d+), #pending-token:\s*(\d+)"
)
P_THR_RE = re.compile(r"input throughput \(token/s\):\s*([\d.]+)")
D_RE = re.compile(
    r"#running-req:\s*(\d+), #full token:\s*(\d+), full token usage:\s*([\d.]+), "
    r"#swa token:\s*(\d+), swa token usage:\s*([\d.]+), cuda graph: \w+, "
    r"gen throughput \(token/s\):\s*([\d.]+), #queue-req:\s*(\d+)"
)

EPOCH = datetime.datetime(2000, 1, 1)
def parse_ts(s):
    return int((datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S") - EPOCH).total_seconds())

def new_bucket():
    return dict(
        p_lines=0, p_newseq=0, p_newtok=0, p_cached=0,
        p_queue_max=0, p_pending_max=0, p_thr_sum=0.0, p_thr_n=0,
        d_lines=0, d_run_sum=0, d_run_n=0, d_run_max=0,
        d_thr_sum=0.0, d_queue_max=0,
        d_fullu_sum=0.0, d_fullu_max=0.0, d_swau_sum=0.0, d_swau_max=0.0,
        d_fulltok_sum=0, d_fulltok_n=0,
        dp_run=defaultdict(lambda: [0, 0]),
    )

buckets = defaultdict(new_bucket)
heat = defaultdict(lambda: defaultdict(lambda: [0, 0]))
per_dp = defaultdict(lambda: dict(
    p_lines=0, p_newtok=0, p_cached=0, p_newseq=0,
    d_lines=0, d_run_sum=0, d_run_n=0, d_run_max=0, d_thr_sum=0.0,
))

t_min, t_max = None, None
p_total_lines = d_total_lines = 0
bad = 0

with open(SRC, errors="ignore") as fh:
    for line in fh:
        m = TS_RE.search(line)
        if not m:
            bad += 1
            continue
        t = parse_ts(m.group(1))
        dp = int(m.group(2))
        if t_min is None or t < t_min: t_min = t
        if t_max is None or t > t_max: t_max = t
        b = buckets[t // BW * BW]
        s = per_dp[dp]
        if "Prefill batch" in line:
            pm = P_RE.search(line)
            if not pm:
                bad += 1
                continue
            newseq, newtok, cached = int(pm.group(1)), int(pm.group(2)), int(pm.group(3))
            queue, pending = int(pm.group(7)), int(pm.group(8))
            p_total_lines += 1
            b["p_lines"] += 1
            b["p_newseq"] += newseq
            b["p_newtok"] += newtok
            b["p_cached"] += cached
            if queue > b["p_queue_max"]: b["p_queue_max"] = queue
            if pending > b["p_pending_max"]: b["p_pending_max"] = pending
            tm = P_THR_RE.search(line)
            if tm:
                b["p_thr_sum"] += float(tm.group(1)); b["p_thr_n"] += 1
            s["p_lines"] += 1; s["p_newseq"] += newseq
            s["p_newtok"] += newtok; s["p_cached"] += cached
        else:
            dm = D_RE.search(line)
            if not dm:
                bad += 1
                continue
            run = int(dm.group(1)); fulltok = int(dm.group(2))
            fullu = float(dm.group(3)); swau = float(dm.group(5))
            thr = float(dm.group(6)); queue = int(dm.group(7))
            d_total_lines += 1
            b["d_lines"] += 1
            b["d_run_sum"] += run; b["d_run_n"] += 1
            if run > b["d_run_max"]: b["d_run_max"] = run
            b["d_thr_sum"] += thr
            if queue > b["d_queue_max"]: b["d_queue_max"] = queue
            b["d_fullu_sum"] += fullu
            if fullu > b["d_fullu_max"]: b["d_fullu_max"] = fullu
            b["d_swau_sum"] += swau
            if swau > b["d_swau_max"]: b["d_swau_max"] = swau
            b["d_fulltok_sum"] += fulltok; b["d_fulltok_n"] += 1
            r = b["dp_run"][dp]; r[0] += run; r[1] += 1
            h = heat[t // HEAT_BW * HEAT_BW][dp]; h[0] += run; h[1] += 1
            s["d_lines"] += 1; s["d_run_sum"] += run; s["d_run_n"] += 1
            if run > s["d_run_max"]: s["d_run_max"] = run
            s["d_thr_sum"] += thr

print(f"parsed prefill={p_total_lines} decode={d_total_lines} bad={bad}")
if bad > (p_total_lines + d_total_lines) * 0.01:
    print("WARNING: >1% lines failed to parse -- check the regexes against the log format!")
t0 = t_min

# ---------- timeseries CSV ----------
rows = []
for t in sorted(buckets):
    b = buckets[t]
    rel = t - t0
    d_run_avg = b["d_run_sum"] / b["d_run_n"] if b["d_run_n"] else 0.0
    agg_run = sum(v[0] / v[1] for v in b["dp_run"].values() if v[1])
    n_dp_active = sum(1 for v in b["dp_run"].values() if v[1])
    gen_cluster = b["d_run_sum"] / BW  # exact if decode_log_interval=1
    p_thr_avg = b["p_thr_sum"] / b["p_thr_n"] if b["p_thr_n"] else 0.0
    rows.append(dict(
        rel_s=rel,
        ts=(EPOCH + datetime.timedelta(seconds=t)).strftime("%m-%d %H:%M"),
        p_lines=b["p_lines"], p_newseq=b["p_newseq"], p_newtok=b["p_newtok"],
        p_cached=b["p_cached"], p_queue_max=b["p_queue_max"], p_pending_max=b["p_pending_max"],
        p_thr_avg=round(p_thr_avg, 1),
        d_lines=b["d_lines"], d_run_avg=round(d_run_avg, 2), d_run_max=b["d_run_max"],
        agg_decode_reqs=round(agg_run, 1), n_dp_active=n_dp_active,
        gen_toks_cluster=round(gen_cluster, 0), d_queue_max=b["d_queue_max"],
        full_usage_avg=round(b["d_fullu_sum"] / b["d_lines"], 4) if b["d_lines"] else 0,
        full_usage_max=round(b["d_fullu_max"], 4),
        swa_usage_avg=round(b["d_swau_sum"] / b["d_lines"], 4) if b["d_lines"] else 0,
        full_tok_avg=round(b["d_fulltok_sum"] / b["d_fulltok_n"], 0) if b["d_fulltok_n"] else 0,
    ))

cols = list(rows[0].keys())
with open(os.path.join(BASE, "timeseries_60s.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

with open(os.path.join(BASE, "timeseries_60s.txt"), "w") as fh:
    hdr = (f"{'rel_s':>7} {'time':>12} | {'PFbat':>6} {'newseq':>6} {'newtok':>9} {'cached':>10} {'PFq':>4} | "
           f"{'DECbat':>7} {'run/DP':>7} {'aggReq':>7} {'gen tok/s':>10} {'DECq':>5} {'fullU':>6} {'swaU':>6}")
    fh.write(hdr + "\n" + "-" * len(hdr) + "\n")
    for r in rows:
        fh.write(f"{r['rel_s']:>7} {r['ts']:>12} | {r['p_lines']:>6} {r['p_newseq']:>6} {r['p_newtok']:>9} "
                 f"{r['p_cached']:>10} {r['p_queue_max']:>4} | {r['d_lines']:>7} {r['d_run_avg']:>7.1f} "
                 f"{r['agg_decode_reqs']:>7.0f} {r['gen_toks_cluster']:>10.0f} {r['d_queue_max']:>5} "
                 f"{r['full_usage_avg']:>6.3f} {r['swa_usage_avg']:>6.3f}\n")

# ---------- per-DP stats ----------
with open(os.path.join(BASE, "per_dp_stats.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["dp", "prefill_lines", "prefill_newseq", "prefill_newtok", "prefill_cached",
                "cache_hit_rate", "decode_lines", "decode_run_avg", "decode_run_max", "gen_thr_avg"])
    for dp in sorted(per_dp):
        s = per_dp[dp]
        tot = s["p_newtok"] + s["p_cached"]
        w.writerow([dp, s["p_lines"], s["p_newseq"], s["p_newtok"], s["p_cached"],
                    round(s["p_cached"] / tot, 4) if tot else 0,
                    s["d_lines"],
                    round(s["d_run_sum"] / s["d_run_n"], 2) if s["d_run_n"] else 0,
                    s["d_run_max"],
                    round(s["d_thr_sum"] / s["d_lines"], 1) if s["d_lines"] else 0])

# ---------- heatmap CSV ----------
dps = sorted(per_dp)
heat_ts = sorted(heat)
with open(os.path.join(BASE, "dp_heatmap.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["rel_s"] + [f"DP{d}" for d in dps])
    for t in heat_ts:
        row = [t - t0]
        for d in dps:
            v = heat[t][d]
            row.append(round(v[0] / v[1], 1) if v[1] else "")
        w.writerow(row)

# ---------- summary ----------
total_newtok = sum(s["p_newtok"] for s in per_dp.values())
total_cached = sum(s["p_cached"] for s in per_dp.values())
run_avgs = [s["d_run_sum"] / s["d_run_n"] for s in per_dp.values() if s["d_run_n"]]
summary = dict(
    t_start=(EPOCH + datetime.timedelta(seconds=t_min)).strftime("%Y-%m-%d %H:%M:%S"),
    t_end=(EPOCH + datetime.timedelta(seconds=t_max)).strftime("%Y-%m-%d %H:%M:%S"),
    time_span_s=t_max - t_min,
    time_span_h=round((t_max - t_min) / 3600, 2),
    prefill_lines=p_total_lines, decode_lines=d_total_lines, bad_lines=bad,
    prefill_newseq_total=sum(s["p_newseq"] for s in per_dp.values()),
    prefill_newtok_total=total_newtok, prefill_cached_total=total_cached,
    cache_hit_rate=round(total_cached / (total_newtok + total_cached), 4) if total_newtok + total_cached else 0,
    decode_run_max=max(s["d_run_max"] for s in per_dp.values()),
    per_dp_run_avg_min=round(min(run_avgs), 2) if run_avgs else 0,
    per_dp_run_avg_max=round(max(run_avgs), 2) if run_avgs else 0,
)
with open(os.path.join(BASE, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))

# ---------- plots ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

xs_h = [r["rel_s"] / 3600 for r in rows]
CAP = args.concurrency_cap

def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, name), dpi=110)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(xs_h, [r["agg_decode_reqs"] for r in rows], lw=0.7, color="tab:blue",
        label="cluster active decode reqs (sum of per-DP avg)")
if CAP:
    ax.axhline(CAP, color="red", ls="--", lw=0.8, label=f"{CAP} concurrency cap")
ax.set_xlabel("hours since start"); ax.set_ylabel("requests")
ax.set_title("Cluster-wide active decode requests over full run")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "01_decode_concurrency.png")

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(xs_h, [r["gen_toks_cluster"] for r in rows], lw=0.6, color="tab:green")
ax.set_xlabel("hours since start"); ax.set_ylabel("tok/s")
ax.set_title("Cluster decode throughput (token count per window / bucket)")
ax.grid(alpha=0.3)
save(fig, "02_decode_throughput.png")

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(xs_h, [r["p_cached"] / 1e6 for r in rows], lw=0.6, color="tab:orange", label="cached tok/min (M)")
ax.plot(xs_h, [r["p_newtok"] / 1e6 for r in rows], lw=0.6, color="tab:red", label="new tok/min (M)")
ax.set_xlabel("hours since start"); ax.set_ylabel("tokens per minute (millions)")
ax.set_title("Prefill token volume: cache-hit vs newly computed")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "03_prefill_tokens.png")

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(xs_h, [r["p_queue_max"] for r in rows], lw=0.7, color="tab:purple", label="prefill queue max")
ax.plot(xs_h, [r["d_queue_max"] for r in rows], lw=0.7, color="tab:brown", label="decode queue max")
ax.set_xlabel("hours since start"); ax.set_ylabel("queued reqs (max in window)")
ax.set_title("Server-side queueing over full run")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "04_queues.png")

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(xs_h, [r["full_usage_avg"] for r in rows], lw=0.7, color="tab:blue", label="full token usage (avg)")
ax.plot(xs_h, [r["full_usage_max"] for r in rows], lw=0.5, color="tab:blue", alpha=0.4, label="full token usage (max)")
ax.plot(xs_h, [r["swa_usage_avg"] for r in rows], lw=0.7, color="tab:orange", label="swa token usage (avg)")
ax.set_xlabel("hours since start"); ax.set_ylabel("usage fraction")
ax.set_title("KV cache utilization")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "05_kv_usage.png")

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(xs_h, [r["p_newseq"] for r in rows], lw=0.6, color="tab:cyan")
ax.set_xlabel("hours since start"); ax.set_ylabel("prefill #new-seq per minute")
ax.set_title("New prefill sequences (turn starts) per minute")
ax.grid(alpha=0.3)
save(fig, "06_new_seq_rate.png")

mat = np.full((len(dps), len(heat_ts)), np.nan)
for j, t in enumerate(heat_ts):
    for i, d in enumerate(dps):
        v = heat[t][d]
        if v[1]:
            mat[i, j] = v[0] / v[1]
fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis",
               extent=[(heat_ts[0] - t0) / 3600, (heat_ts[-1] - t0 + HEAT_BW) / 3600, -0.5, len(dps) - 0.5])
ax.set_yticks(range(len(dps))); ax.set_yticklabels([f"DP{d}" for d in dps], fontsize=7)
ax.set_xlabel("hours since start"); ax.set_title(f"Per-DP avg decode running-req ({HEAT_BW//60}-min buckets)")
fig.colorbar(im, ax=ax, label="running-req")
save(fig, "07_dp_heatmap.png")

fig, ax = plt.subplots(figsize=(14, 4.5))
ys = [r["full_tok_avg"] / r["d_run_avg"] / 1000 if r["d_run_avg"] else 0 for r in rows]
ax.plot(xs_h, ys, lw=0.6, color="tab:red")
ax.set_xlabel("hours since start"); ax.set_ylabel("avg KV tokens per running req (K)")
ax.set_title("Average context length per active decode request")
ax.grid(alpha=0.3)
save(fig, "08_ctx_per_req.png")

print("plots written to", PLOTS)
