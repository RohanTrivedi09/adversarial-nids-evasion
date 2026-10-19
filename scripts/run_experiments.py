"""Full experiment: 3 training regimes x 2 attacks x eps grid x N seeds.

Usage:
    python scripts/run_experiments.py                 # 5 seeds (default)
    python scripts/run_experiments.py --seeds 1 --epochs 3   # quick smoke run

Outputs:
    results/results.csv   one row per (seed, model, attack, eps)
    results/summary.csv   mean +/- std over seeds
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nids_adv.attacks import fgsm, perturb_attacks_only, pgd  # noqa: E402
from nids_adv.data import load_nsl_kdd  # noqa: E402
from nids_adv.metrics import evaluate  # noqa: E402
from nids_adv.train import (train_baseline, train_fgsm_static_at,  # noqa: E402
                            train_pgd_at)

EPS_GRID = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--eps-train", type=float, default=0.1)
    ap.add_argument("--pgd-steps", type=int, default=20)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = load_nsl_kdd(args.data_dir)
    print(f"train {d.X_train.shape}  test {d.X_test.shape}  "
          f"attackable features {int(d.cont_mask.sum())}/{d.X_train.shape[1]}")

    rows = []
    for seed in range(args.seeds):
        t0 = time.time()
        models = {
            "baseline": train_baseline(d.X_train, d.y_train, seed=seed, epochs=args.epochs),
            "fgsm_static_at": train_fgsm_static_at(d.X_train, d.y_train, d.cont_mask,
                                                   args.eps_train, seed=seed, epochs=args.epochs),
            "pgd_at": train_pgd_at(d.X_train, d.y_train, d.cont_mask, args.eps_train,
                                   seed=seed, epochs=args.epochs),
        }
        for name, m in models.items():
            for atk_name, atk in (("fgsm", fgsm), ("pgd", pgd)):
                for eps in EPS_GRID:
                    kw = {"eps": eps, "mask": d.cont_mask}
                    if atk_name == "pgd":
                        kw.update(steps=args.pgd_steps, rng=np.random.default_rng(seed))
                    X_adv = perturb_attacks_only(atk, m, d.X_test, d.y_test, **kw)
                    r = evaluate(d.y_test, m.predict(X_adv))
                    rows.append({"seed": seed, "model": name, "attack": atk_name,
                                 "eps": eps, **r})
        print(f"seed {seed} done in {time.time() - t0:.1f}s")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "results.csv"), index=False)
    metrics = ["accuracy", "detection_rate", "fpr", "f1", "evasion_rate"]
    summ = df.groupby(["model", "attack", "eps"])[metrics].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    print(summ.reset_index()[["model", "attack", "eps", "detection_rate_mean",
                              "fpr_mean", "f1_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
