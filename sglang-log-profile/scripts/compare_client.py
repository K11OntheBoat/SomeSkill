#!/usr/bin/env python3
"""Client-side PK of two (or more) benchmark result JSONs.

Compares per-turn TTFT and decode-streaming-time distributions, plus per-turn
decode speed (output_len / decode_stream_time) which isolates decode kernel
performance from queueing/prefill effects.

Usage:
  compare_client.py --outdir <pk_dir> \
      --run flashinfer=/path/a.json --run megamoe=/path/b.json
Outputs: client_compare.json, plots/pk07-pk09.
"""
import argparse, json, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--run", action="append", required=True, help="NAME=RESULT_JSON")
args = ap.parse_args()

BASE = os.path.abspath(args.outdir)
PLOTS = os.path.join(BASE, "plots")
os.makedirs(PLOTS, exist_ok=True)

PALETTE = ["tab:blue", "tab:red", "tab:green", "tab:purple"]
SRCS, COLORS = {}, {}
for i, spec in enumerate(args.run):
    name, src = spec.split("=", 1)
    SRCS[name] = src
    COLORS[name] = PALETTE[i % len(PALETTE)]

runs = {}
for name, src in SRCS.items():
    d = json.load(open(src))
    valid = [(t, i or 0, o or 0, sum(x)) for t, i, o, x in
             zip(d["ttfts"], d["input_lens"], d["output_lens"], d["itls"]) if t and t > 0]
    ttft = [v[0] for v in valid]
    outlen = [v[2] for v in valid]
    dect = [v[3] for v in valid]
    dspeed = [o / t for o, t in zip(outlen, dect) if t > 1.0 and o >= 32]
    runs[name] = dict(d=d, ttft=ttft, dect=dect, dspeed=dspeed)

def pct(v, p):
    return float(np.percentile(v, p))

out = {}
for name, r in runs.items():
    d = r["d"]
    e = dict(
        duration_s=round(d["duration"], 0),
        completed=d["completed"],
        output_tok_throughput=round(d["output_throughput"], 1) if d.get("output_throughput") else None,
        ttft_mean=round(statistics.mean(r["ttft"]), 1),
        ttft_median=round(statistics.median(r["ttft"]), 1),
        ttft_p99=round(d["p99_ttft_ms"] / 1000, 1) if d.get("p99_ttft_ms") else None,
        decode_stream_mean=round(statistics.mean(r["dect"]), 1),
        decode_stream_median=round(statistics.median(r["dect"]), 1),
        ttft_req_hours=round(sum(r["ttft"]) / 3600),
        decode_req_hours=round(sum(r["dect"]) / 3600),
        ttft_share=round(sum(r["ttft"]) / (sum(r["ttft"]) + sum(r["dect"])), 3),
        per_turn_decode_speed_median=round(statistics.median(r["dspeed"]), 1),
        per_turn_decode_speed_p10=round(pct(r["dspeed"], 10), 1),
        per_turn_decode_speed_p90=round(pct(r["dspeed"], 90), 1),
    )
    if isinstance(d.get("s_decode_clean"), (int, float)):
        e["decode_speed_clean"] = round(d["s_decode_clean"], 2)
    if d.get("n_itls_total"):
        e["itl_preempt_pct"] = round(d["n_itls_preempt"] / d["n_itls_total"] * 100, 2)
    out[name] = e
with open(os.path.join(BASE, "client_compare.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(out, indent=2))

def save(fig, fname):
    fig.tight_layout(); fig.savefig(os.path.join(PLOTS, fname), dpi=110); plt.close(fig)

# pk07: TTFT distributions
fig, ax = plt.subplots(figsize=(10, 4.5))
hi = max(pct(r["ttft"], 99) for r in runs.values())
bins = np.linspace(0, hi * 1.2, 100)
for name, r in runs.items():
    ax.hist(r["ttft"], bins=bins, alpha=0.5, color=COLORS[name],
            label=f"{name} (median {out[name]['ttft_median']}s)")
ax.set_xlabel("TTFT (s)"); ax.set_ylabel("# turns")
ax.set_title("PK: client TTFT distribution")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "pk07_ttft_hist.png")

# pk08: per-turn decode speed distributions
fig, ax = plt.subplots(figsize=(10, 4.5))
hi = max(pct(r["dspeed"], 99) for r in runs.values())
bins = np.linspace(0, hi * 1.2, 100)
for name, r in runs.items():
    ax.hist(r["dspeed"], bins=bins, alpha=0.5, color=COLORS[name], density=True,
            label=f"{name} (median {out[name]['per_turn_decode_speed_median']} tok/s)")
ax.set_xlabel("per-turn decode speed (tok/s)"); ax.set_ylabel("density")
ax.set_title("PK: per-turn decode speed (output_len / decode stream time, output>=32 tok)")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "pk08_decode_speed_hist.png")

# pk09: latency budget bars
fig, ax = plt.subplots(figsize=(8, 4.5))
names = list(runs)
ttft_h = [out[n]["ttft_req_hours"] for n in names]
dec_h = [out[n]["decode_req_hours"] for n in names]
x = np.arange(len(names))
ax.bar(x, ttft_h, 0.5, color="tab:orange", label="TTFT (wait+prefill)")
ax.bar(x, dec_h, 0.5, bottom=ttft_h, color="tab:green", label="decode streaming")
for i, n in enumerate(names):
    ax.text(i, ttft_h[i] / 2, f"{ttft_h[i]} h\n({out[n]['ttft_share']*100:.0f}%)", ha="center", va="center", fontsize=9)
    ax.text(i, ttft_h[i] + dec_h[i] / 2, f"{dec_h[i]} h", ha="center", va="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("total request-hours across all turns")
ax.set_title("PK: where did all the latency go?")
ax.legend(); ax.grid(alpha=0.3, axis="y")
save(fig, "pk09_latency_budget.png")
print("plots pk07-pk09 written")
