#!/usr/bin/env python3
"""Client-side analysis of a benchmark result JSON (FastDeploy bench format:
ttfts / input_lens / output_lens / itls / duration / completed ...).

Cross-validates the server-side log analysis and decomposes per-request
latency into TTFT (admission/prefill wait) vs decode streaming time.

Usage: analyze_client.py --json <result.json> --outdir <run_dir>
Outputs: client_summary.json, plots/13-15.
"""
import argparse, json, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True, help="benchmark result JSON")
ap.add_argument("--outdir", required=True)
args = ap.parse_args()

BASE = os.path.abspath(args.outdir)
os.makedirs(os.path.join(BASE, "plots"), exist_ok=True)
d = json.load(open(args.json))

tt = d["ttfts"]; il = d["input_lens"]; ol = d["output_lens"]; itls = d["itls"]
dur = d["duration"]
valid = [(t, i or 0, o or 0, sum(x)) for t, i, o, x in zip(tt, il, ol, itls) if t and t > 0]
ttft = [v[0] for v in valid]
inlen = [v[1] for v in valid]
dectime = [v[3] for v in valid]

tt_sum = sum(ttft); dec_sum = sum(dectime)
summary = dict(
    duration_s=dur,
    completed=d["completed"],
    failed=sum(1 for e in d.get("errors", []) if e),
    sessions=d.get("num_prompts"),
    turns_per_session=round(d["completed"] / d["num_prompts"], 1) if d.get("num_prompts") else None,
    total_input_tokens=d.get("total_input_tokens"),
    total_output_tokens=d.get("total_output_tokens"),
    total_token_throughput=round(d["total_token_throughput"]) if d.get("total_token_throughput") else None,
    output_token_throughput=round(d["output_throughput"], 1) if d.get("output_throughput") else None,
    mean_ttft_s=round(statistics.mean(ttft), 1),
    median_ttft_s=round(statistics.median(ttft), 1),
    p99_ttft_s=round(d["p99_ttft_ms"] / 1000, 1) if d.get("p99_ttft_ms") else None,
    mean_decode_stream_s=round(statistics.mean(dectime), 1),
    median_decode_stream_s=round(statistics.median(dectime), 1),
    ttft_req_hours=round(tt_sum / 3600),
    decode_req_hours=round(dec_sum / 3600),
    ttft_share_of_latency=round(tt_sum / (tt_sum + dec_sum), 3),
    avg_sessions_in_ttft=round(tt_sum / dur),
    avg_sessions_in_decode=round(dec_sum / dur),
)
# optional custom fields (not always present)
if d.get("n_itls_total"):
    summary["itl_preempt_gt500ms_pct"] = round(d["n_itls_preempt"] / d["n_itls_total"] * 100, 2)
if isinstance(d.get("s_decode_clean"), (int, float)):
    summary["decode_speed_clean_tok_s"] = round(d["s_decode_clean"], 2)

with open(os.path.join(BASE, "client_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
print(json.dumps(summary, indent=2))

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(BASE, "plots", name), dpi=110); plt.close(fig)

# 13. TTFT histogram
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.hist(ttft, bins=100, color="tab:blue", alpha=0.85)
ax.set_xlabel("TTFT (s)"); ax.set_ylabel("# requests (turns)")
ax.set_title(f"Client TTFT distribution (mean {summary['mean_ttft_s']}s, median {summary['median_ttft_s']}s, n={len(ttft)})")
ax.grid(alpha=0.3)
save(fig, "13_client_ttft_hist.png")

# 14. TTFT vs input length
fig, ax = plt.subplots(figsize=(10, 4.5))
hb = ax.hexbin([x / 1000 for x in inlen], ttft, gridsize=60, cmap="viridis", mincnt=1)
ax.set_xlabel("input length (K tokens)"); ax.set_ylabel("TTFT (s)")
ax.set_title("TTFT vs input length per turn")
fig.colorbar(hb, ax=ax, label="# turns")
save(fig, "14_ttft_vs_inputlen.png")

# 15. latency composition per request, sorted by total
tot = sorted(zip(ttft, dectime), key=lambda p: p[0] + p[1])
xs = [i / len(tot) * 100 for i in range(len(tot))]
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.fill_between(xs, 0, [p[0] for p in tot], color="tab:orange", alpha=0.8, label="TTFT (wait+prefill)")
ax.fill_between(xs, [p[0] for p in tot], [p[0] + p[1] for p in tot], color="tab:green", alpha=0.8, label="decode streaming")
ax.set_xlabel("requests sorted by total latency (percentile)"); ax.set_ylabel("seconds")
ax.set_title(f"Per-request latency composition: TTFT = {summary['ttft_share_of_latency']*100:.0f}% of total request time")
ax.legend(); ax.grid(alpha=0.3)
save(fig, "15_latency_composition.png")
print("plots 13-15 written")
