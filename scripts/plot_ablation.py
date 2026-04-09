#!/usr/bin/env python3
"""Generate ablation figures: violation-vs-seed, CVaR bar, Max-Loss bar.

Usage (inside `solar` conda env):
    python scripts/plot_ablation.py [--data_mode strict]
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_ablation(data_mode: str = "strict"):
    p = Path(f"results/{data_mode}/ablation_quantile.json")
    with open(p) as f:
        d = json.load(f)
    seeds = d.get("_seeds", [0, 1, 2, 42])
    algos = ["DDPG", "DR3L_full", "DR3L_Quantile"]

    data = {}
    for algo in algos:
        rets, cvars, maxls, viols = [], [], [], []
        for seed in seeds:
            sk, rk = f"seed_{seed}", f"{algo}_seed{seed}"
            if sk in d and rk in d[sk]:
                t = d[sk][rk]["test"]
                rets.append(t.get("episode_returns_mean", float("nan")))
                cvars.append(t.get("episode_cvars_mean", float("nan")))
                maxls.append(t.get("episode_max_losses_mean", float("nan")))
                viols.append(
                    t.get("violation_any_rate_mean",
                          t.get("episode_viol_any_mean", float("nan")))
                )
        data[algo] = dict(
            seeds=seeds, rets=rets, cvars=cvars, maxls=maxls,
            viols=[100 * v for v in viols],
        )
    return data, seeds


def mean(xs):
    return sum(xs) / len(xs) if xs else 0

def std(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) if xs else 0


COLORS = {"DDPG": "#2196F3", "DR3L_full": "#E53935", "DR3L_Quantile": "#4CAF50"}
MARKERS = {"DDPG": "o", "DR3L_full": "s", "DR3L_Quantile": "D"}
LABELS = {"DDPG": "DDPG", "DR3L_full": "DR3L-Full (N=51, adaptive)",
          "DR3L_Quantile": "DR3L-Quantile (N=8, fixed)"}


def plot_violation_vs_seed(data, seeds, out_dir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = list(range(len(seeds)))
    for algo in ["DDPG", "DR3L_full", "DR3L_Quantile"]:
        d = data[algo]
        ax.plot(x, d["viols"], marker=MARKERS[algo], color=COLORS[algo],
                lw=2.2, ms=9, label=LABELS[algo])
        for i, y in enumerate(d["viols"]):
            ax.annotate(f"{y:.1f}%", (x[i], y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8.5,
                        color=COLORS[algo])
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds], fontsize=12)
    ax.set_xlabel("Random Seed", fontsize=13)
    ax.set_ylabel("Violation Rate (%)", fontsize=13)
    ax.set_ylim(-3, 105)
    ax.set_title("Ablation: Violation Rate vs. Seed", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out_dir / "ablation_violation_vs_seed.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def _bar_chart(data, metric_key, ylabel, title, filename, out_dir):
    algos = ["DDPG", "DR3L_full", "DR3L_Quantile"]
    means = [mean(data[a][metric_key]) for a in algos]
    stds = [std(data[a][metric_key]) for a in algos]
    x = np.arange(len(algos))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(x, means, yerr=stds, capsize=6, width=0.55,
                  color=[COLORS[a] for a in algos], edgecolor="white", lw=1.2)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                f"{m:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[a] for a in algos], fontsize=9.5, rotation=12,
                       ha="right")
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = out_dir / filename
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_mode", default="strict")
    args = ap.parse_args()

    data, seeds = load_ablation(args.data_mode)
    out_dir = Path(f"results/{args.data_mode}")

    # Print summary
    print("=" * 70)
    print("Ablation Summary")
    print("=" * 70)
    for algo in ["DDPG", "DR3L_full", "DR3L_Quantile"]:
        d = data[algo]
        print(f"\n{algo}:")
        print(f"  Return   : {mean(d['rets']):.1f} ± {std(d['rets']):.1f}")
        print(f"  CVaR     : {mean(d['cvars']):.3f} ± {std(d['cvars']):.3f}")
        print(f"  Max Loss : {mean(d['maxls']):.3f} ± {std(d['maxls']):.3f}")
        print(f"  Viol%    : {mean(d['viols']):.1f}% ± {std(d['viols']):.1f}%"
              f"  per-seed: {[f'{v:.1f}%' for v in d['viols']]}")

    # Plots
    plot_violation_vs_seed(data, seeds, out_dir)
    _bar_chart(data, "cvars", "CVaR₀.₁", "Ablation: CVaR (mean ± std)",
               "ablation_cvar_bar.png", out_dir)
    _bar_chart(data, "maxls", "Max Loss", "Ablation: Max Loss (mean ± std)",
               "ablation_max_loss_bar.png", out_dir)

    # Save JSON summary
    summary = {}
    for algo in ["DDPG", "DR3L_full", "DR3L_Quantile"]:
        d = data[algo]
        summary[algo] = {
            "return_mean": mean(d["rets"]),   "return_std": std(d["rets"]),
            "cvar_mean": mean(d["cvars"]),     "cvar_std": std(d["cvars"]),
            "max_loss_mean": mean(d["maxls"]), "max_loss_std": std(d["maxls"]),
            "violation_mean": mean(d["viols"]),
            "violation_std": std(d["viols"]),
            "violation_per_seed": d["viols"],
        }
    sp = out_dir / "ablation_summary.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {sp}")


if __name__ == "__main__":
    main()
