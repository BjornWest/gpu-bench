#!/usr/bin/env python3
"""
Parse bundled InferenceX CSVs in reference/inferencex/ and emit
throughput-optimized picks for our matrix models on 8×B200 and 8×B300.

Selection rules:
- Single-node only: TP <= 8 AND Is Multinode = false AND Hardware Key does
  NOT contain "dynamo-" (which routes across nodes).
- Group by (model, hardware, Hardware Key, TP) so b200_trt and b200_trt_mtp
  are kept distinct.
- Aggregate node throughput = OutputTPS/GPU × 8 (the 8 GPUs of the node are
  utilised either by one TP=8 instance or several smaller parallel instances
  routed by our proxy — same total decode capacity).
- Total node concurrency at a row = per-instance Concurrency × (8 / TP).
- Two picks per (model, HW):
    a) Within-napkin: best aggregate where total node conc <= peak-slots cap.
    b) Tested ceiling: best aggregate from any tested row (informational).
- If the tested ceiling on (model, HW) sits below the napkin cap, the parser
  fits a power-law to the top 2 tested points and projects throughput at
  the napkin cap. A 0.85 discount factor is applied for conservatism.

Context degradation: InferenceX measures at ISL=8192. Our audit profile
averages ~30k. Throughput scales sub-linearly with context due to attention
cost + KV-cache movement. Conservative multipliers:
  MLA (DeepSeek, GLM, Kimi): 0.5x
  GQA (MiniMax):             0.4x
Re-benchmark on rack before locking — these are first-order estimates.
"""

from __future__ import annotations

import csv
from math import log
from pathlib import Path

STATS_DIR = Path(__file__).resolve().parents[1] / "reference" / "inferencex"

# Napkin VRAM peak-context (60k) slot ceiling on the whole 8-GPU node.
# Source: deployment_scripts/neoclouds/verda/HARDWARE.md tables.
PEAK_SLOTS_B200 = {
    "DeepSeek-R1-0528": 237,   # use V3.2 family number
    "DeepSeek-V4-Pro": 237,    # MLA-class, similar weight footprint to V3.2
    "GLM-5": 181,
    "Kimi-K2.5": 190,
    "MiniMax-M2.5": 169,       # M2.5/M2.7 share architecture
}
PEAK_SLOTS_B300 = {
    "DeepSeek-R1-0528": 393,
    "DeepSeek-V4-Pro": 393,
    "GLM-5": 305,
    "Kimi-K2.5": 347,
    "MiniMax-M2.5": 257,
}

ATTENTION = {
    "DeepSeek-R1-0528": "MLA",
    "DeepSeek-V4-Pro": "MLA",
    "GLM-5": "MLA",
    "Kimi-K2.5": "MLA",
    "MiniMax-M2.5": "GQA",
}

CTX_DEGRADE = {"MLA": 0.5, "GQA": 0.4}

EXTRAPOLATION_DISCOUNT = 0.85  # be 15% pessimistic on projected throughput

# Per-model KV-cache budget and per-token KV cost (8×B200 / 8×B300).
# Source: deployment_scripts/neoclouds/verda/HARDWARE.md tables. Used to emit calc.py config inputs.
KV_PER_TOKEN_KB = {
    "DeepSeek-R1-0528": 70,
    "DeepSeek-V4-Pro": 70,
    "GLM-5": 88,
    "Kimi-K2.5": 70,
    "MiniMax-M2.5": 124,
}
KV_BUDGET_GB_B200 = {
    "DeepSeek-R1-0528": 997,
    "DeepSeek-V4-Pro": 997,
    "GLM-5": 954,
    "Kimi-K2.5": 800,
    "MiniMax-M2.5": 1262,
}
KV_BUDGET_GB_B300 = {
    "DeepSeek-R1-0528": 1653,
    "DeepSeek-V4-Pro": 1653,
    "GLM-5": 1610,
    "Kimi-K2.5": 1456,
    "MiniMax-M2.5": 1918,
}
# Display label → config.yaml name mapping (M2.5 stands in for M2.7 since
# they share architecture).
CONFIG_NAME = {
    "DeepSeek-R1-0528": "DeepSeek V3.2",
    "DeepSeek-V4-Pro": "DeepSeek V4 Pro",
    "GLM-5": "GLM-5",
    "Kimi-K2.5": "Kimi K2.5",
    "MiniMax-M2.5": "MiniMax M2.7",
}


def parse_csv(path: Path):
    with path.open() as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        yield row


def is_single_node(row) -> bool:
    if row["Hardware"] not in ("b200", "b300"):
        return False
    if int(row["TP"]) > 8:
        return False
    if row["Is Multinode"].lower() == "true":
        return False
    if "dynamo" in row["Hardware Key"]:
        return False
    return True


def node_tps(row) -> float:
    return float(row["Output Throughput/GPU (tok/s)"]) * 8


def node_conc(row) -> int:
    """Total node concurrency = per-instance Concurrency × (8 / TP)."""
    return int(row["Concurrency"]) * (8 // int(row["TP"]))


def per_user_tps(row) -> float:
    return float(row["Mean Interactivity (tok/s/user)"])


def estimate_at_cap(rows, target_conc) -> tuple[float, str] | tuple[None, None]:
    """Estimate node throughput at target_conc, given tested rows.

    Three cases:
      - target_conc bracketed by tested points: linear interpolation (no discount)
      - target_conc above all tested: power-law from last 2 points × discount
      - target_conc below all tested: nearest tested point (no projection useful)

    Returns (estimate, method) or (None, None) if insufficient data."""
    if len(rows) < 2:
        return None, None
    pts = sorted(rows, key=node_conc)

    # Linear interpolation if bracketed
    for i in range(len(pts) - 1):
        c1, c2 = node_conc(pts[i]), node_conc(pts[i + 1])
        if c1 <= target_conc <= c2 and c1 != c2:
            t1, t2 = node_tps(pts[i]), node_tps(pts[i + 1])
            est = t1 + (t2 - t1) * (target_conc - c1) / (c2 - c1)
            return est, "interp"

    # Extrapolation upward if target > tested max
    if target_conc > node_conc(pts[-1]):
        p1, p2 = pts[-2], pts[-1]
        c1, c2 = node_conc(p1), node_conc(p2)
        t1, t2 = node_tps(p1), node_tps(p2)
        if c1 == c2 or t1 <= 0 or t2 <= 0:
            return None, None
        b = log(t2 / t1) / log(c2 / c1)
        projected = t2 * (target_conc / c2) ** b * EXTRAPOLATION_DISCOUNT
        return projected, "extrap"

    return None, None


def best_pick(rows_by_group, cap: int):
    """Across all (HardwareKey, TP) groups, find the row with max node TPS
    where node conc <= cap. Returns (row, hwkey, tp) or None."""
    best = None
    for (hwkey, tp), rows in rows_by_group.items():
        for r in rows:
            if node_conc(r) > cap:
                continue
            if best is None or node_tps(r) > node_tps(best[0]):
                best = (r, hwkey, tp)
    return best


def best_tested(rows_by_group):
    """Across all groups, find the row with max node TPS (any concurrency)."""
    best = None
    for (hwkey, tp), rows in rows_by_group.items():
        for r in rows:
            if best is None or node_tps(r) > node_tps(best[0]):
                best = (r, hwkey, tp)
    return best


def emit_config_yaml(hw: str, kv_budget: dict, recs: dict):
    """Emit ready-to-paste config.yaml snippets in the new schema."""
    print()
    print(f"  Ready-to-paste config snippets for 8×{hw.upper()}:")
    print(f"  {'─' * 70}")
    for model_label, rec in recs.items():
        if rec is None:
            continue
        cfg_name = CONFIG_NAME.get(model_label, model_label)
        kv_kb = KV_PER_TOKEN_KB.get(model_label, 100)
        kv_gb = kv_budget.get(model_label, 1000)
        attn = ATTENTION.get(model_label, "MLA")
        print(f"    - name: {cfg_name}")
        print(f"      attention: {attn}")
        print(f"      per_user_tps_at_ref: {rec['per_user_at_pick']:.0f}")
        print(f"      max_aggregate_at_ref: {rec['max_aggregate_tested']:.0f}")
        print(f"      kv_per_token_kb: {kv_kb}")
        print(f"      kv_budget_gb: {kv_gb}")


def emit_table(hw: str, model_data: dict, peak_slots: dict):
    """One row per model for the given hardware."""
    print(f"\n8×{hw.upper()} — single-node single-or-parallel-instance picks")
    print(f"{'─' * 116}")
    print(f"{'Model':<18} {'Pick (cfg)':<28} {'NodeTPS':>8} {'tok/s/u':>8} "
          f"{'TestedMax':>10} {'Napkin':>7} {'Projected':>10} {'30k-adj':>9}")
    print(f"{'─' * 116}")

    recommendations = {}

    for model, groups in sorted(model_data.items()):
        if not groups:
            print(f"  {model:<18} (no single-node {hw} data)")
            continue
        cap = peak_slots.get(model, 200)
        attn = ATTENTION.get(model, "MLA")
        degrade = CTX_DEGRADE[attn]

        # Within-napkin best
        wn = best_pick(groups, cap)
        # Overall tested best
        bt = best_tested(groups)

        if wn is None:
            print(f"  {model:<18} no row within napkin {cap}; tested max only")
            wn = bt

        pick_row, hwkey, tp = wn
        cfg = f"{hwkey.split('_', 1)[1]} TP={tp} c={int(pick_row['Concurrency'])}({node_conc(pick_row)} node)"
        n_tps = node_tps(pick_row)
        pu = per_user_tps(pick_row)

        # Tested max across all groups
        tested_max_conc = max(
            node_conc(r) for rows in groups.values() for r in rows
        )

        # Estimate at napkin cap. If the winning group's tested points
        # bracket the cap, interpolate; if they're all below, extrapolate
        # (power-law, discounted). The estimate is shown for transparency and
        # used only when it exceeds the tested pick.
        group_rows = groups[(hwkey, tp)]
        proj_val, proj_method = estimate_at_cap(group_rows, cap)
        used_projection = proj_val is not None and proj_val > n_tps
        adj_base = proj_val if used_projection else n_tps
        adj_30k = adj_base * degrade

        proj_str = "—"
        if proj_val is not None:
            proj_str = f"{proj_val:.0f}({proj_method[:6]})"

        print(f"  {model:<18} {cfg:<28} {n_tps:>8.0f} {pu:>8.1f} "
              f"{tested_max_conc:>10} {cap:>7} {proj_str:>10} "
              f"{adj_30k:>9.0f}")

        # Peak aggregate observed across all groups (decode-ceiling at ref ctx)
        max_agg_tested = max(
            node_tps(r) for rows in groups.values() for r in rows
        )

        recommendations[model] = {
            "config": cfg,
            "node_tps_within_napkin": int(n_tps),
            "tested_max_node_conc": tested_max_conc,
            "napkin_peak": cap,
            "projected_at_napkin": int(proj_val) if proj_val is not None else None,
            "projection_method": proj_method,
            "used_projection": used_projection,
            "throughput_30k_adj": int(adj_30k),
            "per_user_tps": round(pu, 1),
            "per_user_at_pick": pu,
            "max_aggregate_tested": max_agg_tested,
        }
    return recommendations


def main():
    # Group rows by (model, hardware) then by (hwkey, tp)
    all_data: dict[tuple[str, str], dict[tuple[str, int], list]] = {}
    for csv_path in sorted(STATS_DIR.glob("*.csv")):
        model = csv_path.stem.replace("InferenceX_", "").replace("_interactivity", "")
        for row in parse_csv(csv_path):
            if not is_single_node(row):
                continue
            mh_key = (model, row["Hardware"])
            grp_key = (row["Hardware Key"], int(row["TP"]))
            all_data.setdefault(mh_key, {}).setdefault(grp_key, []).append(row)

    # Per-hardware tables
    for hw, peak, kv_budget in [
        ("b200", PEAK_SLOTS_B200, KV_BUDGET_GB_B200),
        ("b300", PEAK_SLOTS_B300, KV_BUDGET_GB_B300),
    ]:
        model_data = {
            m: groups for (m, h), groups in all_data.items() if h == hw
        }
        for m in peak:
            model_data.setdefault(m, {})
        recs = emit_table(hw, model_data, peak)
        emit_config_yaml(hw, kv_budget, recs)
    print()

    print("Legend:")
    print("  Pick (cfg):   chosen Hardware-Key variant, TP, per-instance concurrency,")
    print("                and total node concurrency in parentheses (= per-inst × 8/TP).")
    print("  NodeTPS:      aggregate node output tok/s at the chosen tested point.")
    print("  TestedMax:    highest total node concurrency InferenceX measured.")
    print("  Napkin:       peak-context (60k) node slot ceiling from HARDWARE.md.")
    print("  Projected:    estimate at napkin cap, method in parentheses:")
    print("                  (interp) — napkin sits between tested points, linear interp")
    print(f"                  (extrap) — napkin > tested max, power-law × "
          f"{EXTRAPOLATION_DISCOUNT} discount")
    print("                Only used when it exceeds tested NodeTPS.")
    print("  30k-adj:      best estimate × context-degradation factor "
          f"(MLA={CTX_DEGRADE['MLA']}, GQA={CTX_DEGRADE['GQA']}).")
    print()
    print("Notes:")
    print("  - DeepSeek R1 B300 has only sglang single-node data (poor). Realistic")
    print("    V3.2 B300 throughput with TRT-LLM MTP should be at least the B200")
    print("    trt_mtp result × ~1.3 (VRAM headroom). Treat 'b300 R1' here as a")
    print("    floor only.")
    print("  - DeepSeek V4 Pro B300: no single-node data. Use B200 × ~1.2 as floor.")
    print("  - Mistral Medium 3.5 128B, Qwen 3.6, Nemotron, Gemma 4: no InferenceX")
    print("    coverage. Benchmark on rack before locking config.yaml.")


if __name__ == "__main__":
    main()
