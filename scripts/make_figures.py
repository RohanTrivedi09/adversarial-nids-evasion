"""Build all figures from results/summary.csv (run run_experiments.py first)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

OUT = "figures"
LABELS = {"baseline": "Baseline (no defense)",
          "fgsm_static_at": "Static FGSM adv. training",
          "pgd_at": "PGD adv. training"}
COLORS = {"baseline": "#6b7280", "fgsm_static_at": "#d97706", "pgd_at": "#2563eb"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200})


def main():
    os.makedirs(OUT, exist_ok=True)
    s = pd.read_csv("results/summary.csv")

    # 1. Detection rate under PGD vs eps
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for m in LABELS:
        d = s[(s.model == m) & (s.attack == "pgd")]
        ax.plot(d.eps, 100 * d.detection_rate_mean, "o-", color=COLORS[m], label=LABELS[m], ms=4)
        ax.fill_between(d.eps, 100 * (d.detection_rate_mean - d.detection_rate_std),
                        100 * (d.detection_rate_mean + d.detection_rate_std),
                        color=COLORS[m], alpha=0.15)
    ax.set_xlabel(r"Perturbation budget $\epsilon$ ($L_\infty$, scaled features)")
    ax.set_ylabel("Detection rate under PGD-20 (%)")
    ax.set_ylim(0, 80)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Evasion robustness (mean ± std, 5 seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/dr_vs_eps_pgd.png")
    plt.close(fig)

    # 2. FGSM vs PGD: how much single-step evaluation overstates robustness
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.9), sharey=True)
    for ax, m in zip(axes, LABELS):
        f = s[(s.model == m) & (s.attack == "fgsm")]
        p = s[(s.model == m) & (s.attack == "pgd")]
        ax.plot(f.eps, 100 * f.detection_rate_mean, "s--", color=COLORS[m], label="FGSM", ms=4)
        ax.plot(p.eps, 100 * p.detection_rate_mean, "o-", color=COLORS[m], label="PGD-20", ms=4)
        ax.fill_between(p.eps, 100 * p.detection_rate_mean.values,
                        100 * f.detection_rate_mean.values, color=COLORS[m], alpha=0.15)
        ax.set_title(LABELS[m], fontsize=9)
        ax.set_xlabel(r"$\epsilon$")
    axes[0].set_ylabel("Detection rate (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Shaded gap = robustness that FGSM-only evaluation over-reports", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fgsm_vs_pgd_gap.png")
    plt.close(fig)

    # 3. Clean-traffic trade-off
    c = s[(s.eps == 0) & (s.attack == "pgd")].set_index("model").loc[list(LABELS)]
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = range(len(c))
    w = 0.26
    for k, (col, lab) in enumerate([("detection_rate_mean", "Detection rate"),
                                    ("fpr_mean", "False-positive rate"),
                                    ("f1_mean", "F1")]):
        ax.bar([i + (k - 1) * w for i in x], 100 * c[col], w, label=lab,
               color=["#1f2937", "#dc2626", "#059669"][k])
    ax.set_xticks(list(x), [LABELS[m].replace(" adv. training", "\nadv. training") for m in c.index], fontsize=8)
    ax.set_ylabel("%")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(f"{OUT}/clean_tradeoff.png")
    plt.close(fig)
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
