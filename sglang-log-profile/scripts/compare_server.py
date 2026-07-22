#!/usr/bin/env python3
"""Server-side PK of two (or more) runs on the same benchmark.

Reads each run's timeseries_60s.csv (active window only) and produces:
  - plots/pk01..pk06 : overlay time-series comparisons
  - bands_compare.csv: per-concurrency-band wall-clock/token/per-req-speed table
  - server_compare.json: headline numbers side by side

Usage:
  compare_server.py --outdir <pk_dir> \
      --run flashinfer=/path/run_a=19200 --run megamoe=/path/run_b=19800
  (--run NAME=RUN_DIR=ACTIVE_WINDOW_SECONDS, repeatable)
Optional: --cap 768 (concurrency cap reference line)
"""
import argparse, csv, json, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--run", action="append", required=True,
                help="NAME=RUN_DIR=ACTIVE_WINDOW_SECONDS")
ap.add_argument("--cap", type=int, default=None)
args = ap.parse_args()

BASE = os.path.abspath(args.outdir)
PLOTS = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)

PALETTE = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
RUNS = {}
for i, spec in enumerate(args.run):
    name, rdir, lim = spec.split("=")
    RUNS[name] = dict(dir=rdir, lim=int(lim), color=PALETTE[i % len(PALETTE)])

data = {}
for name, cfg in RUNS.items():
    rows = [r for r in csv.DictReader(open(os.path.join(cfg["dir"], "timeseries_60s.csv")))
            if 0 <= int(r["rel_s"]) <= cfg["lim"]]
    data[name] = rows

def col(name, key, cast=float):
    return [cast(r[key]) for r in data[name]]

def xs(name):
    return [int(r["rel_s"]) / 3600 for r in data[name]]

def save(fig, fname):
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, fname), dpi=110); plt.close(fig)

# ---- pk01: active decode concurrency ----
fig, ax = plt.subplots(figsize=(14, 4.5))
for name, cfg in RUNS.items():
    ax.plot(xs(name), col(name, "agg_decode_reqs"), lw=0.9, color=cfg["color"], label=name)
if args.cap:
    ax.axhline(args.cap, color="gray", ls="--", lw=0.8)
ax.set_xlabel("hours since start"); ax.set_ylabel("active decode requests")
ax.set_title("PK: active decode concurrency (faster drain = faster decode)")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "pk01_concurrency.png")

# ---- pk02: cluster decode throughput ----
fig, ax = plt.subplots(figsize=(14, 4.5))
for name, cfg in RUNS.items():
    ax.plot(xs(name), col(name, "gen_toks_cluster"), lw=0.7, color=cfg["color"], alpha=0.8, label=name)
ax.set_xlabel("hours since start"); ax.set_ylabel("tok/s")
ax.set_title("PK: cluster decode throughput (exact token count / 60s)")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "pk02_decode_throughput.png")

# ---- pk03: per-request decode speed ----
fig, ax = plt.subplots(figsize=(14, 4.5))
for name, cfg in RUNS.items():
    g = col(name, "gen_toks_cluster"); a = col(name, "agg_decode_reqs")
    ys = [gi / ai if ai > 2 else float("nan") for gi, ai in zip(g, a)]
    ax.plot(xs(name), ys, lw=0.8, color=cfg["color"], label=name)
ax.set_xlabel("hours since start"); ax.set_ylabel("tok/s per request")
ax.set_title("PK: per-request decode speed (cluster tok/s / active reqs)")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "pk03_per_req_speed.png")

# ---- pk04: per-request speed vs concurrency (cleanest kernel comparison) ----
fig, ax = plt.subplots(figsize=(10, 5))
for name, cfg in RUNS.items():
    g = col(name, "gen_toks_cluster"); a = col(name, "agg_decode_reqs")
    pts = [(ai, gi / ai) for gi, ai in zip(g, a) if ai > 2]
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=6, alpha=0.4, color=cfg["color"], label=name)
ax.set_xlabel("active decode requests (cluster)"); ax.set_ylabel("tok/s per request")
ax.set_title("PK: per-request decode speed vs concurrency")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "pk04_speed_vs_concurrency.png")

# ---- pk05: prefill input throughput ----
fig, ax = plt.subplots(figsize=(14, 4.5))
for name, cfg in RUNS.items():
    ys = [v if v > 0 else float("nan") for v in col(name, "p_thr_avg")]
    ax.plot(xs(name), ys, lw=0.6, color=cfg["color"], alpha=0.8, label=name)
ax.set_xlabel("hours since start"); ax.set_ylabel("input throughput (tok/s, per-line avg)")
ax.set_title("PK: prefill input throughput (from Prefill batch log lines)")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "pk05_prefill_throughput.png")

# ---- pk06: concurrency duration curve ----
fig, ax = plt.subplots(figsize=(8, 4.5))
for name, cfg in RUNS.items():
    levels = sorted(col(name, "agg_decode_reqs"))
    ax.plot(levels, np.linspace(0, 100, len(levels)), lw=1.2, color=cfg["color"], label=name)
ax.set_xlabel("active decode requests"); ax.set_ylabel("% of wall-clock below this level")
ax.set_title("PK: concurrency duration curve (straggler profile)")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "pk06_concurrency_cdf.png")

# ---- bands table ----
cap = args.cap or int(max(max(col(n, "agg_decode_reqs")) for n in RUNS)) + 1
bands = [(700, cap + 1), (400, 700), (200, 400), (100, 200), (50, 100), (10, 50), (0, 10)]
band_rows = []
for lo, hi in bands:
    row = dict(band=f"{lo}-{hi}")
    for name in RUNS:
        rows = data[name]
        tot_t = len(rows)
        tot_tok = sum(float(r["gen_toks_cluster"]) * 60 for r in rows)
        sel = [r for r in rows if lo <= float(r["agg_decode_reqs"]) < hi]
        if sel:
            row[f"{name}_wall_pct"] = round(len(sel) / tot_t * 100, 1)
            row[f"{name}_tok_pct"] = round(sum(float(r["gen_toks_cluster"]) * 60 for r in sel) / tot_tok * 100, 1)
            row[f"{name}_perreq"] = round(sum(float(r["gen_toks_cluster"]) / float(r["agg_decode_reqs"])
                                              for r in sel if float(r["agg_decode_reqs"]) > 0) / len(sel), 1)
        else:
            row[f"{name}_wall_pct"] = row[f"{name}_tok_pct"] = row[f"{name}_perreq"] = ""
    band_rows.append(row)
with open(os.path.join(BASE, "bands_compare.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(band_rows[0].keys()))
    w.writeheader(); w.writerows(band_rows)

# ---- headline compare ----
out = {}
for name in RUNS:
    rows = data[name]
    thr = [float(r["p_thr_avg"]) for r in rows if float(r["p_thr_avg"]) > 0]
    agg = [float(r["agg_decode_reqs"]) for r in rows]
    out[name] = dict(
        window_s=int(rows[-1]["rel_s"]),
        total_decode_tok_M=round(sum(float(r["gen_toks_cluster"]) * 60 for r in rows) / 1e6, 2),
        peak_cluster_toks=max(float(r["gen_toks_cluster"]) for r in rows),
        peak_concurrency=max(agg),
        prefill_thr_mean=round(statistics.mean(thr)) if thr else None,
        prefill_thr_median=round(statistics.median(thr)) if thr else None,
        prefill_queue_max=max(int(r["p_queue_max"]) for r in rows),
        pending_tok_max=max(int(r["p_pending_max"]) for r in rows),
        kv_full_usage_max=max(float(r["full_usage_max"]) for r in rows),
        wall_below50_pct=round(sum(1 for a in agg if a < 50) / len(agg) * 100, 1),
        last_min_above={th: round(max((int(r["rel_s"]) for r in rows
                                       if float(r["agg_decode_reqs"]) >= th), default=0) / 60)
                        for th in (400, 200, 100, 50)},
    )
with open(os.path.join(BASE, "server_compare.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))
print("plots pk01-pk06 written")
