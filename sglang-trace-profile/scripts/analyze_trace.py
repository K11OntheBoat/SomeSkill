#!/usr/bin/env python3
"""
SGLang DeepSeekV4-Flash Profile Trace Analyzer
===============================================
Analyzes PyTorch Profiler Chrome Trace Format JSON and generates 5 markdown reports:
  01_cpu_stack.md       - CPU call stack per phase/layer/component
  02_gpu_stack.md       - GPU kernel top-10 + category breakdown per phase
  03_cpu_gpu_gap.md     - CPU vs GPU timing gap analysis
  04_bottleneck_analysis.md - System bottleneck identification
  05_extra_analysis.md  - Additional analysis (per-layer variance, MoE, etc.)
"""

import json
import os
import sys
import argparse
import re
from collections import defaultdict, Counter
from bisect import bisect_left, bisect_right


# =============================================================================
# Utility Functions
# =============================================================================

def format_duration(us: float) -> str:
    """Format microseconds to human-readable string."""
    if us < 0:
        return f"-{format_duration(-us)}"
    if us < 1000:
        return f"{us:.1f}us"
    elif us < 1_000_000:
        return f"{us/1000:.2f}ms"
    else:
        return f"{us/1_000_000:.3f}s"


def simplify_kernel_name(name: str) -> str:
    """Simplify kernel name: keep namespace::function, collapse template params to <...>."""
    # Remove 'void ' prefix
    if name.startswith('void '):
        name = name[5:]
    # Handle leading (anonymous namespace)::
    prefix = ''
    while name.startswith('(anonymous namespace)::'):
        prefix = 'anon::'
        name = name[len('(anonymous namespace)::'):]
    # Collapse template parameters - stop at top-level '(' that isn't inside <...>
    result = [prefix] if prefix else []
    depth_angle = 0
    i = 0
    while i < len(name):
        ch = name[i]
        if ch == '<':
            if depth_angle == 0:
                result.append('<...>')
            depth_angle += 1
        elif ch == '>':
            depth_angle -= 1
        elif depth_angle == 0:
            if ch == '(':
                break  # start of function arguments
            result.append(ch)
        i += 1
    simplified = ''.join(result).strip()
    return simplified if simplified else name.split('(')[0].split('<')[0][:80]


COMM_KEYWORDS = ['nccl', 'deep_ep', 'allreduce', 'allgather', 'broadcast',
                 'reduce_scatter', 'send', 'recv', 'internode']


def classify_kernel(name: str) -> str:
    """Classify a GPU kernel into Communication / GEMM / Attention / Other."""
    name_lower = name.lower()
    if any(kw in name_lower for kw in COMM_KEYWORDS):
        return 'Communication'
    if 'flash' in name_lower or 'attention' in name_lower:
        return 'Attention'
    if 'mla' in name_lower and 'gemm' not in name_lower:
        return 'Attention'
    if 'mqa_logits' in name_lower or 'mla_combine' in name_lower:
        return 'Attention'
    if 'gemm' in name_lower or 'nvjet' in name_lower or 'cutlass' in name_lower:
        return 'GEMM'
    if 'quant' in name_lower:
        return 'Quantization'
    if 'norm' in name_lower or 'rmsnorm' in name_lower:
        return 'Norm'
    if 'rope' in name_lower:
        return 'RoPE'
    if 'silu' in name_lower or 'gelu' in name_lower or 'act_and_mul' in name_lower:
        return 'Activation'
    if 'tilelang' in name_lower or 'mhc' in name_lower:
        return 'TileLang/MHC'
    if 'elementwise' in name_lower or 'vectorized' in name_lower:
        return 'Elementwise'
    if 'topk' in name_lower or 'softmax' in name_lower:
        return 'TopK/Softmax'
    return 'Other'


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================

def load_trace(trace_path: str) -> dict:
    """Load Chrome Trace Format JSON file."""
    file_size = os.path.getsize(trace_path)
    print(f"  Loading trace: {trace_path} ({file_size / 1e9:.2f} GB)")
    with open(trace_path, 'r') as f:
        data = json.load(f)
    events = data.get('traceEvents', data if isinstance(data, list) else [])
    print(f"  Total events: {len(events):,}")
    return data


def preprocess_events(data: dict) -> dict:
    """Group events by category, identify main thread, pre-sort key lists."""
    events = data.get('traceEvents', data if isinstance(data, list) else [])

    by_cat = defaultdict(list)
    for e in events:
        cat = e.get('cat', '')
        if cat and e.get('ph') == 'X':
            by_cat[cat].append(e)

    # Find main thread (most python_function events with scheduler/model_runner)
    tid_counts = Counter()
    for e in by_cat['python_function']:
        name = e.get('name', '')
        if 'scheduler.py' in name or 'model_runner' in name:
            tid_counts[e.get('tid')] += 1
    main_tid = tid_counts.most_common(1)[0][0] if tid_counts else None
    print(f"  Main thread tid: {main_tid}")

    # Main-thread python functions, sorted by (ts, -dur)
    py_funcs = sorted(
        [e for e in by_cat['python_function'] if e.get('tid') == main_tid],
        key=lambda e: (e['ts'], -e.get('dur', 0))
    )
    # Pre-extract timestamps for bisect
    py_funcs_ts = [e['ts'] for e in py_funcs]

    # GPU kernels sorted by ts
    kernels = sorted(by_cat.get('kernel', []), key=lambda e: e['ts'])
    kernels_ts = [e['ts'] for e in kernels]

    # GPU user annotations
    gpu_annots = by_cat.get('gpu_user_annotation', [])

    # CUDA runtime
    cuda_runtime = by_cat.get('cuda_runtime', [])

    # GPU memcpy
    gpu_memcpy = by_cat.get('gpu_memcpy', [])

    return {
        'by_cat': by_cat,
        'main_tid': main_tid,
        'py_funcs': py_funcs,
        'py_funcs_ts': py_funcs_ts,
        'kernels': kernels,
        'kernels_ts': kernels_ts,
        'gpu_annots': gpu_annots,
        'cuda_runtime': cuda_runtime,
        'gpu_memcpy': gpu_memcpy,
        'device_props': data.get('deviceProperties', []),
        'dist_info': data.get('distributedInfo', {}),
    }


# =============================================================================
# Phase Classification
# =============================================================================

def classify_phases(processed: dict) -> dict:
    """Classify batches into Extend / Decode-Eager / Decode-CUDAGraph (both CPU & GPU)."""
    py_funcs = processed['py_funcs']
    gpu_annots = processed['gpu_annots']

    # --- CPU-side: find run_batch events and classify ---
    run_batches = [e for e in py_funcs if 'run_batch' in e.get('name', '') and 'scheduler.py' in e.get('name', '')]
    run_batches.sort(key=lambda e: e['ts'])

    extend_batches = []
    decode_eager_batches = []
    decode_graph_batches = []

    for rb in run_batches:
        rb_start = rb['ts']
        rb_end = rb_start + rb['dur']
        # Check child events within this run_batch
        is_extend = False
        is_graph = False
        for e in py_funcs:
            if e['ts'] < rb_start:
                continue
            if e['ts'] > rb_end:
                break
            name = e.get('name', '')
            if '_execute_extend' in name:
                is_extend = True
                break
            if 'decode_cuda_graph_runner' in name and 'execute' in name and 'can_run' not in name:
                is_graph = True
                break

        if is_extend:
            extend_batches.append(rb)
        elif is_graph:
            decode_graph_batches.append(rb)
        else:
            decode_eager_batches.append(rb)

    # --- GPU-side: classify step annotations (use main compute stream) ---
    # Find the main stream: the one with both EXTEND and DECODE steps, most total
    step_annots_all = [e for e in gpu_annots if e.get('name', '').startswith('step[')]
    tid_has_extend = set()
    for s in step_annots_all:
        if 'EXTEND' in s['name']:
            tid_has_extend.add(s.get('tid'))
    # Use the tid that has both extend+decode and the most steps
    main_gpu_tid = None
    best_count = 0
    for tid in tid_has_extend:
        cnt = sum(1 for s in step_annots_all if s.get('tid') == tid)
        if cnt > best_count:
            best_count = cnt
            main_gpu_tid = tid

    step_annots = [e for e in step_annots_all if e.get('tid') == main_gpu_tid] if main_gpu_tid else step_annots_all

    extend_steps = [s for s in step_annots if 'EXTEND' in s['name']]
    decode_steps = [s for s in step_annots if 'DECODE' in s['name']]

    # Auto-detect threshold between CUDAGraph (fast) and Eager (slow)
    if decode_steps:
        sorted_durs = sorted(s['dur'] for s in decode_steps)
        max_gap_idx = 0
        max_gap_ratio = 1.0
        for i in range(len(sorted_durs) - 1):
            if sorted_durs[i] > 0:
                ratio = sorted_durs[i + 1] / sorted_durs[i]
                if ratio > max_gap_ratio:
                    max_gap_ratio = ratio
                    max_gap_idx = i
        if max_gap_ratio > 5:
            decode_threshold = (sorted_durs[max_gap_idx] + sorted_durs[max_gap_idx + 1]) / 2
        else:
            decode_threshold = 100_000
        print(f"  GPU main stream tid: {main_gpu_tid}, decode threshold: {format_duration(decode_threshold)} (gap: {max_gap_ratio:.1f}x)")
    else:
        decode_threshold = 100_000

    decode_eager_steps = [s for s in decode_steps if s['dur'] >= decode_threshold]
    decode_graph_steps = [s for s in decode_steps if s['dur'] < decode_threshold]

    print(f"  CPU batches: Extend={len(extend_batches)}, Decode-Eager={len(decode_eager_batches)}, Decode-Graph={len(decode_graph_batches)}")
    print(f"  GPU steps:   Extend={len(extend_steps)}, Decode-Eager={len(decode_eager_steps)}, Decode-Graph={len(decode_graph_steps)}")

    return {
        'extend_batches': extend_batches,
        'decode_eager_batches': decode_eager_batches,
        'decode_graph_batches': decode_graph_batches,
        'extend_steps': extend_steps,
        'decode_eager_steps': decode_eager_steps,
        'decode_graph_steps': decode_graph_steps,
    }


# =============================================================================
# Report 1: CPU Stack
# =============================================================================

from dataclasses import dataclass, field
from typing import Dict

@dataclass
class CallStackNode:
    name: str
    total_dur_us: float = 0.0
    self_dur_us: float = 0.0
    count: int = 0
    children: Dict[str, 'CallStackNode'] = field(default_factory=dict)


def build_call_tree(py_funcs, py_funcs_ts, start_ts, end_ts, max_depth=15):
    """Build call tree for events within [start_ts, end_ts]."""
    i_start = bisect_left(py_funcs_ts, start_ts)
    i_end = bisect_right(py_funcs_ts, end_ts)

    filtered = []
    for idx in range(i_start, i_end):
        e = py_funcs[idx]
        dur = e.get('dur', 0)
        if dur > 0 and e['ts'] >= start_ts and e['ts'] + dur <= end_ts + 1:
            filtered.append(e)

    root = CallStackNode(name=f"[TOTAL: {format_duration(end_ts - start_ts)}]",
                         total_dur_us=end_ts - start_ts)
    stack = [(root, end_ts)]

    for e in filtered:
        e_start = e['ts']
        e_end = e_start + e['dur']
        # Pop until parent contains this event
        while len(stack) > 1 and e_start >= stack[-1][1]:
            stack.pop()
        if len(stack) > max_depth:
            continue
        parent_node = stack[-1][0]
        name = e.get('name', '?')
        if name not in parent_node.children:
            parent_node.children[name] = CallStackNode(name=name)
        child = parent_node.children[name]
        child.total_dur_us += e['dur']
        child.count += 1
        stack.append((child, e_end))

    # Compute self time
    def _compute_self(node):
        children_total = sum(c.total_dur_us for c in node.children.values())
        node.self_dur_us = max(0, node.total_dur_us - children_total)
        for c in node.children.values():
            _compute_self(c)
    _compute_self(root)
    return root


def format_call_tree_md(node, total_time, min_pct=0.5, indent_prefix="", is_last=True, depth=0, lines=None):
    """Format call tree as markdown-compatible text with tree characters."""
    if lines is None:
        lines = []

    if depth == 0:
        # Root
        dur_str = format_duration(node.total_dur_us)
        lines.append(f"[{dur_str:>10} 100.0%] {node.name}")
        children = sorted(node.children.values(), key=lambda c: -c.total_dur_us)
        for i, child in enumerate(children):
            pct = child.total_dur_us / total_time * 100
            if pct < min_pct:
                continue
            is_last_child = (i == len(children) - 1) or all(
                c.total_dur_us / total_time * 100 < min_pct for c in children[i+1:]
            )
            format_call_tree_md(child, total_time, min_pct, "", is_last_child, 1, lines)
    else:
        connector = "└── " if is_last else "├── "
        dur_str = format_duration(node.total_dur_us)
        pct = node.total_dur_us / total_time * 100
        count_str = f" (x{node.count})" if node.count > 1 else ""
        lines.append(f"{indent_prefix}{connector}[{dur_str:>10} {pct:>5.1f}%] {node.name}{count_str}")

        extension = "    " if is_last else "│   "
        new_prefix = indent_prefix + extension
        children = sorted(node.children.values(), key=lambda c: -c.total_dur_us)
        visible_children = [c for c in children if c.total_dur_us / total_time * 100 >= min_pct]
        for i, child in enumerate(visible_children):
            is_last_child = (i == len(visible_children) - 1)
            format_call_tree_md(child, total_time, min_pct, new_prefix, is_last_child, depth + 1, lines)

    return lines

def extract_layer_components(py_funcs, py_funcs_ts, layer_event):
    """Extract 4 main components within a DecoderLayer by timestamp order."""
    dl_start = layer_event['ts']
    dl_end = dl_start + layer_event['dur']

    # Find events within this layer's time window using bisect
    i_start = bisect_left(py_funcs_ts, dl_start)
    i_end = bisect_right(py_funcs_ts, dl_end)

    # Extract layer_id from name like "nn.Module: DeepseekV4DecoderLayer_4"
    layer_name = layer_event.get('name', '')
    layer_id_match = re.search(r'_(\d+)$', layer_name.split(':')[-1].strip())
    layer_id = layer_id_match.group(1) if layer_id_match else '0'

    mhc_events = []
    mqa_event = None
    moe_event = None

    for idx in range(i_start, i_end):
        e = py_funcs[idx]
        if e['ts'] + e.get('dur', 0) > dl_end:
            continue
        name = e.get('name', '')
        if 'mhc_fused_post_pre' in name and 'try_fused' not in name:
            mhc_events.append(e)
        elif name == f'nn.Module: MQALayer_{layer_id}':
            mqa_event = e
        elif name == f'nn.Module: DeepseekV2MoE_{layer_id}':
            moe_event = e

    mhc_events.sort(key=lambda e: e['ts'])

    # Distinguish 1st (pre-attn) vs 2nd (pre-MoE) by position
    if len(mhc_events) >= 2:
        mhc_pre_attn = mhc_events[0]
        mhc_pre_moe = mhc_events[1]
    elif len(mhc_events) == 1:
        mhc_pre_attn = None
        mhc_pre_moe = mhc_events[0]
    else:
        mhc_pre_attn = None
        mhc_pre_moe = None

    return {
        'layer_id': int(layer_id),
        'total_dur': layer_event['dur'],
        'mhc_pre_attn': mhc_pre_attn,
        'mqa': mqa_event,
        'mhc_pre_moe': mhc_pre_moe,
        'moe': moe_event,
    }


def extract_subcomponents(py_funcs, py_funcs_ts, parent_event, patterns):
    """Extract sub-components within a parent event by name matching."""
    if parent_event is None:
        return {}
    p_start = parent_event['ts']
    p_end = p_start + parent_event['dur']
    i_start = bisect_left(py_funcs_ts, p_start)
    i_end = bisect_right(py_funcs_ts, p_end)

    results = {}
    for idx in range(i_start, i_end):
        e = py_funcs[idx]
        if e['ts'] + e.get('dur', 0) > p_end:
            continue
        name = e.get('name', '')
        for pat_key, pat_match in patterns.items():
            if pat_key in results:
                continue  # already found
            if callable(pat_match):
                if pat_match(name):
                    results[pat_key] = e
            elif pat_match in name:
                results[pat_key] = e
    return results


def generate_01_cpu_stack(processed: dict, phases: dict, output_dir: str):
    """Generate Report 1: CPU Stack Analysis."""
    print("\n  [Report 1] CPU Stack Analysis...")
    py_funcs = processed['py_funcs']
    py_funcs_ts = processed['py_funcs_ts']

    lines = []
    lines.append("# CPU Stack 观察 - DeepSeekV4-Flash on SGLang\n")

    # Overview table
    ext = phases['extend_batches']
    eag = phases['decode_eager_batches']
    grp = phases['decode_graph_batches']
    lines.append("## Overview\n")
    lines.append("| Phase | Batches | Avg run_batch Duration |")
    lines.append("|-------|---------|----------------------|")
    if ext:
        avg_ext = sum(b['dur'] for b in ext) / len(ext)
        lines.append(f"| Extend (Prefill) | {len(ext)} | {format_duration(avg_ext)} |")
    if eag:
        avg_eag = sum(b['dur'] for b in eag) / len(eag)
        lines.append(f"| Decode-Eager | {len(eag)} | {format_duration(avg_eag)} |")
    if grp:
        avg_grp = sum(b['dur'] for b in grp) / len(grp)
        lines.append(f"| Decode-CUDAGraph | {len(grp)} | {format_duration(avg_grp)} |")
    lines.append("")

    # Helper: analyze one phase
    def analyze_phase(batches, phase_name, include_per_layer=True):
        lines.append(f"---\n\n## {phase_name}\n")
        if not batches:
            lines.append("(无数据)\n")
            return

        # Find model forward within run_batch
        total_forward_dur = 0
        forward_count = 0
        for rb in batches:
            rb_start = rb['ts']
            rb_end = rb_start + rb['dur']
            i_s = bisect_left(py_funcs_ts, rb_start)
            i_e = bisect_right(py_funcs_ts, rb_end)
            for idx in range(i_s, i_e):
                e = py_funcs[idx]
                if 'forward_batch_generation' in e.get('name', ''):
                    total_forward_dur += e['dur']
                    forward_count += 1
                    break

        if forward_count > 0:
            avg_fwd = total_forward_dur / forward_count
            lines.append(f"### Model Forward 总耗时: avg {format_duration(avg_fwd)} ({forward_count} batches)\n")

        # --- Full call tree (use longest batch as sample) ---
        # Skip the first/last batches when choosing the representative sample:
        # torch.profiler's Python-function tracking can miss frames on warmup /
        # cooldown batches (e.g. Inductor/Dynamo first-compile or teardown),
        # producing a call tree where children of `_execute_extend` show up as
        # siblings of it under `execute`. Middle batches are steady-state.
        # batches come from classify_phases sorted by ts, so trim head/tail.
        candidates = batches[1:-1] if len(batches) >= 3 else batches
        rep_batch = max(candidates, key=lambda b: b['dur'])
        rb_start = rep_batch['ts']
        rb_end = rb_start + rep_batch['dur']
        total_time = rep_batch['dur']

        lines.append(f"### 调用栈 (最长 batch, 总耗时 {format_duration(total_time)})\n")
        lines.append("```")
        tree = build_call_tree(py_funcs, py_funcs_ts, rb_start, rb_end, max_depth=12)
        tree_lines = format_call_tree_md(tree, total_time, min_pct=0.5)
        lines.extend(tree_lines)
        lines.append("```\n")

        if not include_per_layer:
            lines.append(f"*CUDA Graph replay 跳过 Python 层，无 per-layer 事件*\n")
            return

        # Find all DecoderLayer events in this batch
        i_s = bisect_left(py_funcs_ts, rb_start)
        i_e = bisect_right(py_funcs_ts, rb_end)
        layer_events = []
        for idx in range(i_s, i_e):
            e = py_funcs[idx]
            if e.get('name', '').startswith('nn.Module: DeepseekV4DecoderLayer_'):
                layer_events.append(e)

        if not layer_events:
            lines.append("(未找到 DecoderLayer 事件)\n")
            return

        # Extract components for each layer
        layer_data = []
        for le in layer_events:
            comp = extract_layer_components(py_funcs, py_funcs_ts, le)
            layer_data.append(comp)
        layer_data.sort(key=lambda d: d['layer_id'])

        # Per-layer table
        lines.append(f"### Per-Layer 耗时 ({len(layer_data)} layers, 以最长 batch 为例)\n")
        lines.append("| Layer | Total | 1st mhc_fused_post_pre | MQALayer | 2nd mhc_fused_post_pre | DeepseekV2MoE |")
        lines.append("|-------|-------|----------------------|----------|----------------------|---------------|")

        for ld in layer_data:
            total = format_duration(ld['total_dur'])
            mhc1 = format_duration(ld['mhc_pre_attn']['dur']) if ld['mhc_pre_attn'] else "-"
            mqa = format_duration(ld['mqa']['dur']) if ld['mqa'] else "-"
            mhc2 = format_duration(ld['mhc_pre_moe']['dur']) if ld['mhc_pre_moe'] else "-"
            moe = format_duration(ld['moe']['dur']) if ld['moe'] else "-"
            lines.append(f"| {ld['layer_id']} | {total} | {mhc1} | {mqa} | {mhc2} | {moe} |")
        lines.append("")

        # Sub-component detail for a representative middle layer (Layer 4)
        rep_layer = next((ld for ld in layer_data if ld['layer_id'] == 4), layer_data[1] if len(layer_data) > 1 else layer_data[0])

        # MQALayer sub-components — emit one table per requested layer.
        # DeepSeek-V4 alternates attention structure between odd and even
        # layers (odd: MLA, even: C4), so we show Layer 3 alongside Layer 4.
        def _emit_mqa_breakdown(layer_data_entry):
            if not layer_data_entry or not layer_data_entry['mqa']:
                return
            mqa_e = layer_data_entry['mqa']
            mqa_start = mqa_e['ts']
            mqa_end = mqa_start + mqa_e['dur']
            i_s2 = bisect_left(py_funcs_ts, mqa_start)
            i_e2 = bisect_right(py_funcs_ts, mqa_end)

            mqa_subs = defaultdict(float)
            for idx in range(i_s2, i_e2):
                e = py_funcs[idx]
                if e['ts'] + e.get('dur', 0) > mqa_end:
                    continue
                name = e.get('name', '')
                if 'nn.Module: ReplicatedLinear_' in name:
                    mqa_subs['ReplicatedLinear'] += e['dur']
                elif 'nn.Module: ColumnParallelLinear_' in name:
                    mqa_subs['ColumnParallelLinear'] += e['dur']
                elif 'fused_q_norm_rope' in name:
                    mqa_subs['fused_q_norm_rope'] += e['dur']
                elif 'set_swa_key_buffer' in name:
                    mqa_subs['set_swa_key_buffer'] += e['dur']
                elif 'nn.Module: C4Indexer_' in name:
                    mqa_subs['C4Indexer'] += e['dur']
                elif 'nn.Module: RowParallelLinear_' in name:
                    mqa_subs['RowParallelLinear'] += e['dur']
                elif 'nn.Module: RMSNorm_' in name:
                    mqa_subs['RMSNorm'] += e['dur']

            lines.append(f"### MQALayer 子组件 (Layer {layer_data_entry['layer_id']})\n")
            lines.append("| Component | Duration | % of MQALayer |")
            lines.append("|-----------|----------|---------------|")
            mqa_total = mqa_e['dur']
            for comp, dur in sorted(mqa_subs.items(), key=lambda x: -x[1]):
                pct = dur / mqa_total * 100
                lines.append(f"| {comp} | {format_duration(dur)} | {pct:.1f}% |")
            lines.append("")

        # Layer 3 (odd — MLA) then Layer 4 (even — C4). Skip Layer 3 if it
        # coincides with rep_layer to avoid duplicating the same table.
        layer3_entry = next((ld for ld in layer_data if ld['layer_id'] == 3), None)
        if layer3_entry and layer3_entry is not rep_layer:
            _emit_mqa_breakdown(layer3_entry)
        _emit_mqa_breakdown(rep_layer)

        # DeepseekV2MoE sub-components
        if rep_layer['moe']:
            moe_e = rep_layer['moe']
            moe_start = moe_e['ts']
            moe_end = moe_start + moe_e['dur']
            i_s3 = bisect_left(py_funcs_ts, moe_start)
            i_e3 = bisect_right(py_funcs_ts, moe_end)

            moe_subs = defaultdict(float)
            for idx in range(i_s3, i_e3):
                e = py_funcs[idx]
                if e['ts'] + e.get('dur', 0) > moe_end:
                    continue
                name = e.get('name', '')
                if 'nn.Module: MoEGate_' in name:
                    moe_subs['MoEGate'] += e['dur']
                elif 'nn.Module: DeepseekV2MLP_' in name:
                    moe_subs['DeepseekV2MLP (shared expert)'] += e['dur']
                elif 'nn.Module: TopK_' in name:
                    moe_subs['TopK'] += e['dur']
                elif 'nn.Module: DeepEPMoE_' in name:
                    moe_subs['DeepEPMoE'] += e['dur']
                elif 'nn.Module: HashTopK_' in name:
                    moe_subs['HashTopK'] += e['dur']

            lines.append(f"### DeepseekV2MoE 子组件 (Layer {rep_layer['layer_id']})\n")
            lines.append("| Component | Duration | % of MoE |")
            lines.append("|-----------|----------|----------|")
            moe_total = moe_e['dur']
            for comp, dur in sorted(moe_subs.items(), key=lambda x: -x[1]):
                pct = dur / moe_total * 100
                lines.append(f"| {comp} | {format_duration(dur)} | {pct:.1f}% |")
            lines.append("")

        # Layer 3 internal call tree detail (odd layer - different Attention structure)
        layer3_events = [le for le in layer_events if 'DecoderLayer_3' in le.get('name', '')]
        if layer3_events:
            l3 = layer3_events[0]
            lines.append(f"### 单层 DecoderLayer 内部详解 (Layer 3) — MLA Attention\n")
            lines.append("```")
            l3_tree = build_call_tree(py_funcs, py_funcs_ts, l3['ts'], l3['ts'] + l3['dur'], max_depth=8)
            l3_lines = format_call_tree_md(l3_tree, l3['dur'], min_pct=2.0)
            lines.extend(l3_lines)
            lines.append("```\n")

        # Layer 4 internal call tree detail (even layer - C4 Attention structure)
        layer4_events = [le for le in layer_events if 'DecoderLayer_4' in le.get('name', '')]
        if layer4_events:
            l4 = layer4_events[0]
            lines.append(f"### 单层 DecoderLayer 内部详解 (Layer 4) — C4 Attention\n")
            lines.append("```")
            l4_tree = build_call_tree(py_funcs, py_funcs_ts, l4['ts'], l4['ts'] + l4['dur'], max_depth=8)
            l4_lines = format_call_tree_md(l4_tree, l4['dur'], min_pct=2.0)
            lines.extend(l4_lines)
            lines.append("```\n")

    # Generate for each phase
    analyze_phase(ext, "1. Extend (Prefill)", include_per_layer=True)
    analyze_phase(eag, "2. Decode-Eager (无 CUDA Graph)", include_per_layer=True)
    analyze_phase(grp, "3. Decode-CUDAGraph", include_per_layer=False)

    # Write report
    report_path = os.path.join(output_dir, "01_cpu_stack.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  [SAVED] {report_path}")


# =============================================================================
# Report 2: GPU Stack
# =============================================================================

def get_kernels_in_window(kernels, kernels_ts, start, end):
    """Get all kernels within a time window using bisect."""
    i_start = bisect_left(kernels_ts, start)
    i_end = bisect_right(kernels_ts, end)
    return [kernels[i] for i in range(i_start, i_end) if kernels[i]['ts'] + kernels[i].get('dur', 0) <= end]


def generate_02_gpu_stack(processed: dict, phases: dict, output_dir: str):
    """Generate Report 2: GPU Stack Analysis."""
    print("\n  [Report 2] GPU Stack Analysis...")
    kernels = processed['kernels']
    kernels_ts = processed['kernels_ts']

    lines = []
    lines.append("# GPU Stack 观察 - DeepSeekV4-Flash on SGLang\n")

    def analyze_gpu_phase(steps, phase_name):
        lines.append(f"---\n\n## {phase_name}\n")
        if not steps:
            lines.append("(无数据)\n")
            return

        # Collect all kernels across all steps in this phase
        kernel_agg = defaultdict(lambda: {'dur': 0, 'cnt': 0})
        cat_agg = defaultdict(float)
        total_kernel_time = 0

        for step in steps:
            s_start = step['ts']
            s_end = s_start + step['dur']
            step_kernels = get_kernels_in_window(kernels, kernels_ts, s_start, s_end)
            for k in step_kernels:
                name = k.get('name', '')
                dur = k.get('dur', 0)
                kernel_agg[name]['dur'] += dur
                kernel_agg[name]['cnt'] += 1
                cat = classify_kernel(name)
                cat_agg[cat] += dur
                total_kernel_time += dur

        lines.append(f"**{len(steps)} steps, 总 kernel 耗时: {format_duration(total_kernel_time)}, "
                     f"Per-step 平均: {format_duration(total_kernel_time / len(steps))}**\n")

        # Top-10 kernels
        lines.append("### Top-10 Kernels (按总耗时)\n")
        lines.append("| # | Kernel | Total | Count | Avg | Category |")
        lines.append("|---|--------|-------|-------|-----|----------|")
        top10 = sorted(kernel_agg.items(), key=lambda x: -x[1]['dur'])[:10]
        for i, (name, stats) in enumerate(top10, 1):
            short_name = simplify_kernel_name(name)
            avg = stats['dur'] / stats['cnt']
            cat = classify_kernel(name)
            lines.append(f"| {i} | `{short_name}` | {format_duration(stats['dur'])} | {stats['cnt']} | {format_duration(avg)} | {cat} |")
        lines.append("")

        # Category breakdown
        lines.append("### Kernel 分类统计\n")
        lines.append("| Category | Total | % |")
        lines.append("|----------|-------|---|")
        for cat, dur in sorted(cat_agg.items(), key=lambda x: -x[1]):
            pct = dur / total_kernel_time * 100 if total_kernel_time > 0 else 0
            lines.append(f"| {cat} | {format_duration(dur)} | {pct:.1f}% |")
        lines.append("")

        # Summary: Comm vs Compute vs Attn
        comm = cat_agg.get('Communication', 0)
        gemm = cat_agg.get('GEMM', 0)
        attn = cat_agg.get('Attention', 0)
        other = total_kernel_time - comm - gemm - attn
        lines.append("**核心占比:**\n")
        lines.append(f"- 通信 (Communication): {format_duration(comm)} ({comm/total_kernel_time*100:.1f}%)")
        lines.append(f"- GEMM: {format_duration(gemm)} ({gemm/total_kernel_time*100:.1f}%)")
        lines.append(f"- Attention: {format_duration(attn)} ({attn/total_kernel_time*100:.1f}%)")
        lines.append(f"- 其他: {format_duration(other)} ({other/total_kernel_time*100:.1f}%)")
        lines.append("")

    analyze_gpu_phase(phases['extend_steps'], "1. Extend (Prefill)")
    analyze_gpu_phase(phases['decode_eager_steps'], "2. Decode-Eager (无 CUDA Graph)")
    analyze_gpu_phase(phases['decode_graph_steps'], "3. Decode-CUDAGraph")

    # Write report
    report_path = os.path.join(output_dir, "02_gpu_stack.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  [SAVED] {report_path}")


# =============================================================================
# Report 3: CPU-GPU Gap
# =============================================================================

def generate_03_cpu_gpu_gap(processed: dict, phases: dict, output_dir: str):
    """Generate Report 3: CPU-GPU Gap Analysis."""
    print("\n  [Report 3] CPU-GPU Gap Analysis...")
    kernels = processed['kernels']
    kernels_ts = processed['kernels_ts']

    lines = []
    lines.append("# CPU-GPU Gap 分析 - DeepSeekV4-Flash on SGLang\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Phase | CPU Wall (avg) | GPU Wall (avg) | Gap | GPU Kernel Sum (avg) | Kernel Util |")
    lines.append("|-------|---------------|---------------|-----|---------------------|-------------|")

    def compute_phase_gap(cpu_batches, gpu_steps, phase_name):
        """Compute CPU-GPU gap for one phase."""
        if not cpu_batches or not gpu_steps:
            return None

        cpu_avg = sum(b['dur'] for b in cpu_batches) / len(cpu_batches)
        gpu_avg = sum(s['dur'] for s in gpu_steps) / len(gpu_steps)
        gap = cpu_avg - gpu_avg

        # Compute kernel utilization within GPU steps
        total_kernel_sum = 0
        total_comm = 0
        total_compute = 0
        for step in gpu_steps:
            s_start = step['ts']
            s_end = s_start + step['dur']
            step_kernels = get_kernels_in_window(kernels, kernels_ts, s_start, s_end)
            for k in step_kernels:
                dur = k.get('dur', 0)
                total_kernel_sum += dur
                if classify_kernel(k.get('name', '')) == 'Communication':
                    total_comm += dur
                else:
                    total_compute += dur

        avg_kernel_sum = total_kernel_sum / len(gpu_steps)
        kernel_util = avg_kernel_sum / gpu_avg * 100 if gpu_avg > 0 else 0

        lines.append(f"| {phase_name} | {format_duration(cpu_avg)} | {format_duration(gpu_avg)} | "
                     f"{format_duration(gap)} | {format_duration(avg_kernel_sum)} | {kernel_util:.1f}% |")

        return {
            'cpu_avg': cpu_avg, 'gpu_avg': gpu_avg, 'gap': gap,
            'avg_kernel_sum': avg_kernel_sum, 'kernel_util': kernel_util,
            'total_comm': total_comm / len(gpu_steps),
            'total_compute': total_compute / len(gpu_steps),
        }

    ext_gap = compute_phase_gap(phases['extend_batches'], phases['extend_steps'], "Extend (Prefill)")
    eag_gap = compute_phase_gap(phases['decode_eager_batches'], phases['decode_eager_steps'], "Decode-Eager")
    grp_gap = compute_phase_gap(phases['decode_graph_batches'], phases['decode_graph_steps'], "Decode-CUDAGraph")
    lines.append("")

    # Detailed analysis per phase
    def detail_section(gap_info, phase_name):
        if gap_info is None:
            return
        lines.append(f"---\n\n## {phase_name} 详细分析\n")
        lines.append(f"- **CPU wall-clock (run_batch)**: {format_duration(gap_info['cpu_avg'])}")
        lines.append(f"- **GPU wall-clock (step annotation)**: {format_duration(gap_info['gpu_avg'])}")
        lines.append(f"- **CPU-GPU Gap**: {format_duration(gap_info['gap'])}")
        if gap_info['gap'] > 0:
            lines.append(f"  - Gap 为正 → CPU 侧有 {format_duration(gap_info['gap'])} 额外开销 (调度/Python/launch)")
        else:
            lines.append(f"  - Gap 为负 → GPU 排队/执行更慢，CPU 先完成 Python 调用")
        lines.append(f"- **GPU kernel sum (per-step)**: {format_duration(gap_info['avg_kernel_sum'])}")
        lines.append(f"  - 通信: {format_duration(gap_info['total_comm'])} ({gap_info['total_comm']/gap_info['avg_kernel_sum']*100:.1f}%)" if gap_info['avg_kernel_sum'] > 0 else "")
        lines.append(f"  - 计算: {format_duration(gap_info['total_compute'])} ({gap_info['total_compute']/gap_info['avg_kernel_sum']*100:.1f}%)" if gap_info['avg_kernel_sum'] > 0 else "")
        lines.append(f"- **Kernel utilization**: {gap_info['kernel_util']:.1f}% (kernel_sum / gpu_wall)")

        # Bottleneck conclusion
        lines.append(f"\n**瓶颈判断:**\n")
        if gap_info['total_comm'] > gap_info['total_compute']:
            lines.append(f"- 通信占主导 ({gap_info['total_comm']/gap_info['avg_kernel_sum']*100:.1f}% kernel time 是通信)")
            lines.append(f"- DeepEP internode 通信是当前最大瓶颈")
        elif gap_info['kernel_util'] < 50:
            lines.append(f"- GPU 利用率低 ({gap_info['kernel_util']:.1f}%)，大量 GPU idle")
            lines.append(f"- 瓶颈在 CPU launch / scheduling overhead")
        else:
            lines.append(f"- GPU 计算密集型，kernel utilization 高")
        lines.append("")

    detail_section(ext_gap, "Extend (Prefill)")
    detail_section(eag_gap, "Decode-Eager")
    detail_section(grp_gap, "Decode-CUDAGraph")

    # Write report
    report_path = os.path.join(output_dir, "03_cpu_gpu_gap.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  [SAVED] {report_path}")


# =============================================================================
# Report 4: System Bottleneck Analysis
# =============================================================================

def generate_04_bottleneck(processed: dict, phases: dict, output_dir: str):
    """Generate Report 4: System Bottleneck Analysis."""
    print("\n  [Report 4] System Bottleneck Analysis...")
    py_funcs = processed['py_funcs']
    py_funcs_ts = processed['py_funcs_ts']
    kernels = processed['kernels']
    kernels_ts = processed['kernels_ts']
    gpu_annots = processed['gpu_annots']

    lines = []
    lines.append("# 系统瓶颈分析 - DeepSeekV4-Flash on SGLang\n")

    # --- 1. GPU Utilization ---
    lines.append("## 1. GPU 利用率\n")
    if kernels:
        profile_start = kernels[0]['ts']
        profile_end = max(k['ts'] + k.get('dur', 0) for k in kernels)
        profile_wall = profile_end - profile_start
        total_kernel_time = sum(k.get('dur', 0) for k in kernels)
        utilization = total_kernel_time / profile_wall * 100 if profile_wall > 0 else 0

        comm_time = sum(k.get('dur', 0) for k in kernels if classify_kernel(k.get('name', '')) == 'Communication')
        compute_time = total_kernel_time - comm_time

        lines.append(f"- Profile 时间窗口: {format_duration(profile_wall)}")
        lines.append(f"- GPU Kernel 总耗时: {format_duration(total_kernel_time)}")
        lines.append(f"- GPU 利用率 (kernel_sum / wall): **{utilization:.1f}%**")
        lines.append(f"  - 通信 kernel: {format_duration(comm_time)} ({comm_time/total_kernel_time*100:.1f}%)")
        lines.append(f"  - 计算 kernel: {format_duration(compute_time)} ({compute_time/total_kernel_time*100:.1f}%)")
        lines.append(f"- **有效计算利用率**: {compute_time/profile_wall*100:.1f}% (compute_only / wall)")
        lines.append("")

    # --- 2. DeepEP Communication Breakdown ---
    lines.append("## 2. DeepEP 通信详细分解\n")
    deep_ep_kernels = [k for k in kernels if 'deep_ep' in k.get('name', '').lower() or 'internode' in k.get('name', '').lower()]
    if deep_ep_kernels:
        ep_cats = defaultdict(lambda: {'dur': 0, 'cnt': 0})
        for k in deep_ep_kernels:
            name = k.get('name', '').lower()
            if 'notify_dispatch' in name:
                cat = 'notify_dispatch (等待远端就绪)'
            elif 'dispatch' in name:
                cat = 'dispatch (发送token到专家)'
            elif 'combine' in name:
                cat = 'combine (收集专家结果)'
            elif 'cached_notify' in name or 'cached' in name:
                cat = 'cached_notify (缓存通知)'
            else:
                cat = 'other'
            ep_cats[cat]['dur'] += k.get('dur', 0)
            ep_cats[cat]['cnt'] += 1

        total_ep = sum(v['dur'] for v in ep_cats.values())
        lines.append(f"DeepEP 总通信耗时: **{format_duration(total_ep)}**\n")
        lines.append("| Operation | Total | % | Count | Avg |")
        lines.append("|-----------|-------|---|-------|-----|")
        for cat, stats in sorted(ep_cats.items(), key=lambda x: -x[1]['dur']):
            pct = stats['dur'] / total_ep * 100
            avg = stats['dur'] / stats['cnt'] if stats['cnt'] > 0 else 0
            lines.append(f"| {cat} | {format_duration(stats['dur'])} | {pct:.1f}% | {stats['cnt']} | {format_duration(avg)} |")
        lines.append("")

    # --- 3. CUDA Graph Coverage ---
    lines.append("## 3. CUDA Graph 覆盖率\n")
    n_eag = len(phases['decode_eager_batches'])
    n_grp = len(phases['decode_graph_batches'])
    n_total_decode = n_eag + n_grp
    if n_total_decode > 0:
        coverage = n_grp / n_total_decode * 100
        lines.append(f"- Decode 总 batches: {n_total_decode}")
        lines.append(f"- CUDA Graph batches: {n_grp} ({coverage:.1f}%)")
        lines.append(f"- Eager batches: {n_eag} ({100-coverage:.1f}%)")
        if phases['decode_eager_batches'] and phases['decode_graph_batches']:
            avg_eager = sum(b['dur'] for b in phases['decode_eager_batches']) / n_eag
            avg_graph = sum(b['dur'] for b in phases['decode_graph_batches']) / n_grp
            speedup = avg_eager / avg_graph if avg_graph > 0 else 0
            lines.append(f"- Eager avg: {format_duration(avg_eager)} vs CUDAGraph avg: {format_duration(avg_graph)}")
            lines.append(f"- **CUDAGraph 加速比: {speedup:.0f}x**")
        lines.append("")

    # --- 4. Scheduling Overhead ---
    lines.append("## 4. 调度开销\n")
    sched_funcs = ['get_next_batch', 'process_batch_result']
    for func_name in sched_funcs:
        matches = [e for e in py_funcs if func_name in e.get('name', '')]
        if matches:
            total = sum(e['dur'] for e in matches)
            avg = total / len(matches)
            lines.append(f"- `{func_name}`: {len(matches)} 次, avg={format_duration(avg)}, total={format_duration(total)}")
    lines.append("")

    # --- 5. Recommendations ---
    lines.append("## 5. 优化建议\n")
    lines.append("### 优先级 1: 降低 DeepEP 通信开销")
    lines.append("- notify_dispatch 占通信时间最大比例，表示远端节点未就绪的等待时间长")
    lines.append("- 建议: 探索通信-计算 overlap，让 shared expert 计算与 DeepEP dispatch 并行")
    lines.append("")
    lines.append("### 优先级 2: 提升 CUDA Graph 覆盖率")
    lines.append(f"- 当前仅 {n_grp}/{n_total_decode} decode batches 使用 CUDA Graph")
    lines.append("- Eager decode 比 CUDA Graph 慢数十倍，每个 eager batch 都是性能损失")
    lines.append("- 建议: 检查为什么部分 batch size 无法命中 CUDA Graph capture")
    lines.append("")
    lines.append("### 优先级 3: 减少 CPU scheduling overhead")
    lines.append("- get_next_batch + process_batch_result 合计超过 1s")
    lines.append("- 在 CUDA Graph 模式下，CPU scheduling 可能成为瓶颈")
    lines.append("")

    # Write report
    report_path = os.path.join(output_dir, "04_bottleneck_analysis.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  [SAVED] {report_path}")


# =============================================================================
# Report 5: Extra Analysis
# =============================================================================

def generate_05_extra(processed: dict, phases: dict, output_dir: str):
    """Generate Report 5: Additional Analysis."""
    print("\n  [Report 5] Extra Analysis...")
    py_funcs = processed['py_funcs']
    py_funcs_ts = processed['py_funcs_ts']
    kernels = processed['kernels']
    kernels_ts = processed['kernels_ts']

    lines = []
    lines.append("# 补充分析 - DeepSeekV4-Flash on SGLang\n")

    # --- 1. Per-Layer Variance ---
    lines.append("## 1. Per-Layer 耗时方差 (Decode-Eager)\n")
    eag_batches = phases['decode_eager_batches']
    if eag_batches:
        # Collect all DecoderLayer events across all eager batches
        layer_times = defaultdict(list)  # layer_id -> [dur, dur, ...]
        for rb in eag_batches:
            rb_start = rb['ts']
            rb_end = rb_start + rb['dur']
            i_s = bisect_left(py_funcs_ts, rb_start)
            i_e = bisect_right(py_funcs_ts, rb_end)
            for idx in range(i_s, i_e):
                e = py_funcs[idx]
                name = e.get('name', '')
                if name.startswith('nn.Module: DeepseekV4DecoderLayer_'):
                    lid = int(name.split('_')[-1])
                    layer_times[lid].append(e['dur'])

        if layer_times:
            lines.append("| Layer | Avg | Min | Max | Std | Calls |")
            lines.append("|-------|-----|-----|-----|-----|-------|")
            for lid in sorted(layer_times.keys()):
                times = layer_times[lid]
                avg = sum(times) / len(times)
                mn = min(times)
                mx = max(times)
                variance = sum((t - avg) ** 2 for t in times) / len(times)
                std = variance ** 0.5
                lines.append(f"| {lid} | {format_duration(avg)} | {format_duration(mn)} | {format_duration(mx)} | {format_duration(std)} | {len(times)} |")
            lines.append("")

    # --- 2. CUDA Runtime Launch Overhead ---
    lines.append("## 2. CUDA Runtime Launch Overhead\n")
    cuda_rt = processed['cuda_runtime']
    if cuda_rt:
        rt_agg = defaultdict(lambda: {'dur': 0, 'cnt': 0})
        for e in cuda_rt:
            name = e.get('name', '')
            rt_agg[name]['dur'] += e.get('dur', 0)
            rt_agg[name]['cnt'] += 1

        lines.append("| API | Total | Count | Avg |")
        lines.append("|-----|-------|-------|-----|")
        for name, stats in sorted(rt_agg.items(), key=lambda x: -x[1]['dur'])[:10]:
            avg = stats['dur'] / stats['cnt']
            lines.append(f"| `{name}` | {format_duration(stats['dur'])} | {stats['cnt']:,} | {format_duration(avg)} |")
        lines.append("")

    # --- 3. Memory Operations ---
    lines.append("## 3. GPU Memory Operations\n")
    gpu_memcpy = processed['gpu_memcpy']
    if gpu_memcpy:
        mem_agg = defaultdict(lambda: {'dur': 0, 'cnt': 0})
        for e in gpu_memcpy:
            name = e.get('name', '')
            mem_agg[name]['dur'] += e.get('dur', 0)
            mem_agg[name]['cnt'] += 1

        lines.append("| Type | Total | Count |")
        lines.append("|------|-------|-------|")
        for name, stats in sorted(mem_agg.items(), key=lambda x: -x[1]['dur']):
            lines.append(f"| {name} | {format_duration(stats['dur'])} | {stats['cnt']} |")
        lines.append("")

    # --- 4. MoE Internal Component Breakdown (Decode-Eager) ---
    lines.append("## 4. MoE 内部组件耗时 (Decode-Eager 汇总)\n")
    if eag_batches:
        moe_components = defaultdict(lambda: {'dur': 0, 'cnt': 0})
        for rb in eag_batches[:5]:  # sample 5 batches
            rb_start = rb['ts']
            rb_end = rb_start + rb['dur']
            i_s = bisect_left(py_funcs_ts, rb_start)
            i_e = bisect_right(py_funcs_ts, rb_end)
            for idx in range(i_s, i_e):
                e = py_funcs[idx]
                name = e.get('name', '')
                if 'nn.Module: MoEGate_' in name:
                    moe_components['MoEGate']['dur'] += e['dur']
                    moe_components['MoEGate']['cnt'] += 1
                elif 'nn.Module: DeepseekV2MLP_' in name:
                    moe_components['DeepseekV2MLP (shared)']['dur'] += e['dur']
                    moe_components['DeepseekV2MLP (shared)']['cnt'] += 1
                elif 'nn.Module: TopK_' in name:
                    moe_components['TopK']['dur'] += e['dur']
                    moe_components['TopK']['cnt'] += 1
                elif 'nn.Module: DeepEPMoE_' in name:
                    moe_components['DeepEPMoE']['dur'] += e['dur']
                    moe_components['DeepEPMoE']['cnt'] += 1

        total_moe = sum(v['dur'] for v in moe_components.values())
        if total_moe > 0:
            lines.append(f"(基于 {min(5, len(eag_batches))} 个 eager decode batch 汇总)\n")
            lines.append("| Component | Total | % | Count | Avg |")
            lines.append("|-----------|-------|---|-------|-----|")
            for comp, stats in sorted(moe_components.items(), key=lambda x: -x[1]['dur']):
                pct = stats['dur'] / total_moe * 100
                avg = stats['dur'] / stats['cnt'] if stats['cnt'] > 0 else 0
                lines.append(f"| {comp} | {format_duration(stats['dur'])} | {pct:.1f}% | {stats['cnt']} | {format_duration(avg)} |")
            lines.append("")

    # Write report
    report_path = os.path.join(output_dir, "05_extra_analysis.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  [SAVED] {report_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="SGLang DeepSeekV4 Profile Trace Analyzer")
    parser.add_argument('trace_file', help='Path to Chrome Trace JSON file')
    parser.add_argument('--output-dir', default=None, help='Output directory (default: same as trace file)')
    args = parser.parse_args()

    trace_path = os.path.abspath(args.trace_file)
    output_dir = args.output_dir or os.path.dirname(trace_path)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  SGLang DeepSeekV4-Flash Profile Trace Analyzer")
    print("=" * 70)

    # Load
    data = load_trace(trace_path)

    # Preprocess
    print("\n  Preprocessing events...")
    processed = preprocess_events(data)

    # Classify phases
    print("\n  Classifying phases...")
    phases = classify_phases(processed)

    # Generate reports
    generate_01_cpu_stack(processed, phases, output_dir)
    generate_02_gpu_stack(processed, phases, output_dir)
    generate_03_cpu_gpu_gap(processed, phases, output_dir)
    generate_04_bottleneck(processed, phases, output_dir)
    generate_05_extra(processed, phases, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("  分析完成! 报告已保存:")
    print("=" * 70)
    for fname in ['01_cpu_stack.md', '02_gpu_stack.md', '03_cpu_gpu_gap.md',
                  '04_bottleneck_analysis.md', '05_extra_analysis.md']:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            print(f"  📄 {fpath}")
    print()


if __name__ == '__main__':
    main()
