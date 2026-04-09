#!/usr/bin/env python3
"""Generate 'Violation Rate vs Random Seed' figure for the paper."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SEEDS = [0, 1, 2, 42]
OUT_DIR = Path("results/strict")

# ── Load exp5c (DDPG + DR3L_full, old curriculum config) ──
with open(OUT_DIR / "exp5c_multiseed.json") as f:
    d5c = json.load(f)

data = {"DDPG": [], "DR3L_full (exp5c)": []}

for seed in SEEDS:
    sk = f"seed_{seed}"
    for prefix, label in [("DDPG", "DDPG"), ("DR3L_full", "DR3L_full (exp5c)")]:
        key = f"{prefix}_seed{seed}"
        if sk in d5c and key in d5c[sk]:
            t = d5c[sk][key]["test"]
            v = t.get("violation_any_rate_mean",
                       t.get("episode_viol_any_mean", float("nan")))
            data[label].append((seed, v * 100))

# ── Load exp5d (DR3L_full only, stronger phase1 config) ──
p5d = OUT_DIR / "exp5d_stability.json"
if p5d.exists():
    with open(p5d) as f:
        d5d = json.load(f)
    data["DR3L_full (exp5d)"] = []
    for seed in SEEDS:
        sk = f"seed_{seed}"
        key = f"DR3L_full_seed{seed}"
        if sk in d5d and key in d5d[sk]:
            t = d5d[sk][key]["test"]
            v = t.get("violation_any_rate_mean",
                       t.get("episode_viol_any_mean", float("nan")))
            data["DR3L_full (exp5d)"].append((seed, v * 100))

# ── Print stats (for paper table) ──
print("=" * 60)
print("Violation Rate Statistics (per algorithm)")
print("=" * 60)
for algo, pts in data.items():
    if not pts:
        continue
    vals = [p[1] for p in pts]
    print(f"{algo}:")
    print(f"  mean = {np.mean(vals):.2f}%")
    print(f"  std  = {np.std(vals):.2f}%")
    print(f"  per-seed = {[f'{v:.1f}%' for _, v in pts]}")
    print()

# ── Plot ──
fig, ax = plt.subplots(figsize=(7, 4.5))

styles = {
    "DDPG":                dict(marker='o', color='#2196F3', ls='-',  lw=2.2, ms=9, zorder=3),
    "DR3L_full (exp5c)":   dict(marker='s', color='#E53935', ls='-',  lw=2.2, ms=9, zorder=3),
    "DR3L_full (exp5d)":   dict(marker='D', color='#FF9800', ls='--', lw=1.8, ms=7, zorder=2),
}

x_pos = list(range(len(SEEDS)))
x_labels = [str(s) for s in SEEDS]

for algo, pts in data.items():
    if not pts:
        continue
    ys = [p[1] for p in pts]
    ax.plot(x_pos, ys, label=algo, **styles[algo])
    for i, y in enumerate(ys):
        ax.annotate(f"{y:.1f}%", (x_pos[i], y),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=8.5, color=styles[algo]['color'])

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=12)
ax.set_xlabel("Random Seed", fontsize=13)
ax.set_ylabel("Violation Rate (%)", fontsize=13)
ax.set_ylim(-3, 105)
ax.set_title("Constraint Violation Rate vs. Random Seed (strict)", fontsize=14)
ax.legend(fontsize=10.5, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

plt.tight_layout()
out_path = OUT_DIR / "violation_vs_seed.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved: {out_path}")
