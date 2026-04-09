#!/usr/bin/env python3
"""
Cross-domain generalization: Return / CVaR vs domain (raw → light → strict).
Reads exp5b (or configurable) JSON under results/{raw,light,strict}/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DOMAINS = ["raw", "light", "strict"]
METHODS = ["DDPG", "PPO", "DR3L_full"]
STYLE = {
    "DDPG": {"color": "#2196F3", "marker": "o", "ls": "-"},
    "PPO": {"color": "#7B1FA2", "marker": "s", "ls": "-"},
    "DR3L_full": {"color": "#E53935", "marker": "^", "ls": "-"},
}


def load_agents(results_root: Path, json_name: str) -> dict[str, dict[str, dict]]:
    out = {}
    missing = []
    for d in DOMAINS:
        p = results_root / d / json_name
        if not p.exists():
            missing.append(str(p))
            continue
        with open(p, encoding="utf-8") as f:
            out[d] = json.load(f)
    if missing:
        raise FileNotFoundError(
            "Missing result files:\n  " + "\n  ".join(missing)
        )
    return out


def series_for_metric(by_domain: dict, method: str, mean_key: str, std_key: str):
    means, stds = [], []
    for d in DOMAINS:
        block = by_domain[d].get(method, {})
        test = block.get("test", {})
        means.append(float(test.get(mean_key, np.nan)))
        stds.append(float(test.get(std_key, np.nan)))
    return np.array(means, dtype=float), np.array(stds, dtype=float)


def plot_metric(
    by_domain: dict,
    mean_key: str,
    std_key: str,
    ylabel: str,
    title: str,
    outfile: Path,
):
    x = np.arange(len(DOMAINS))
    fig, ax = plt.subplots(figsize=(7.2, 4.5), dpi=150)

    for method in METHODS:
        if not all(method in by_domain[d] for d in DOMAINS):
            continue
        m, s = series_for_metric(by_domain, method, mean_key, std_key)
        st = STYLE[method]
        ax.errorbar(
            x,
            m,
            yerr=s,
            fmt=st["marker"],
            linestyle=st["ls"],
            color=st["color"],
            capsize=4,
            markersize=7,
            linewidth=1.8,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=method.replace("_", " "),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in DOMAINS])
    ax.set_xlabel("Data domain (evaluation)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    ap.add_argument(
        "--json-name",
        default="exp5b_dr3l_phased_wconst_5.json",
        help="JSON file name inside each domain folder",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to results-root",
    )
    args = ap.parse_args()
    out_dir = args.out_dir or args.results_root
    out_dir.mkdir(parents=True, exist_ok=True)

    by_domain = load_agents(args.results_root, args.json_name)

    plot_metric(
        by_domain,
        "episode_returns_mean",
        "episode_returns_std",
        "Return (test, mean ± std over episodes)",
        "Cross-domain generalization — Return",
        out_dir / "generalization_return.png",
    )
    plot_metric(
        by_domain,
        "episode_cvars_mean",
        "episode_cvars_std",
        "CVaR$_{0.1}$ (test, mean ± std over episodes)",
        "Cross-domain generalization — CVaR",
        out_dir / "generalization_cvar.png",
    )


if __name__ == "__main__":
    main()
