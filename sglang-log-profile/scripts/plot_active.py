#!/usr/bin/env python3
"""Zoomed plots over the active benchmark window only (excludes post-run
health-probe tail). Reads timeseries_60s.csv produced by analyze_server.py.

Usage: plot_active.py --dir <run_dir> --end-s <active window seconds> [--concurrency-cap N]
"""
import argparse, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="run dir containing timeseries_60s.csv")
ap.add_argument("--end-s", type=int, required=True, help="end of active window (rel seconds)")
ap.add_argument("--concurrency-cap", type=int, default=None)
args = ap.parse_args()

BASE = os.path.abspath(args.dir)
PLOTS = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)
rows = [r for r in csv.DictReader(open(os.path.join(BASE, "timeseries_60s.csv")))
        if 0 <= int(r["rel_s"]) <= args.end_s]

xs = [int(r["rel_s"]) / 3600 for r in rows]
agg = [float(r["agg_decode_reqs"]) for r in rows]
gen = [float(r["gen_toks_cluster"]) for r in rows]
hours = args.end_s / 3600

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, name), dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(xs, agg, lw=0.9, color="tab:blue")
if args.concurrency_cap:
    ax.axhline(args.concurrency_cap, color="red", ls="--", lw=0.8, label=f"{args.concurrency_cap} cap")
for th, c in [(400, "0.6"), (100, "0.75")]:
    ax.axhline(th, color=c, ls=":", lw=0.7)
ax.set_xlabel("hours since start"); ax.set_ylabel("active decode requests")
ax.set_title(f"Active decode requests, benchmark window only (0-{hours:.1f}h)")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "09_decode_concurrency_active.png")

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(xs, gen, lw=0.9, color="tab:green")
ax.set_xlabel("hours since start"); ax.set_ylabel("tok/s")
ax.set_title("Cluster decode throughput, benchmark window only")
ax.grid(alpha=0.3)
save(fig, "10_decode_throughput_active.png")

fig, ax = plt.subplots(figsize=(14, 4.5))
ys = [g / a if a > 2 else float("nan") for g, a in zip(gen, agg)]
ax.plot(xs, ys, lw=0.9, color="tab:red")
ax.set_xlabel("hours since start"); ax.set_ylabel("tok/s per request")
ax.set_title("Per-request decode speed (cluster tok/s / active reqs)")
ax.grid(alpha=0.3)
save(fig, "11_per_req_speed.png")

levels = sorted(agg)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(levels, np.linspace(0, 100, len(levels)), lw=1.2, color="tab:purple")
ax.set_xlabel("active decode requests"); ax.set_ylabel("% of benchmark wall-clock time below this level")
ax.set_title("Concurrency duration curve (straggler profile)")
ax.grid(alpha=0.3)
save(fig, "12_concurrency_cdf.png")

print("plots 09-12 written to", PLOTS)
