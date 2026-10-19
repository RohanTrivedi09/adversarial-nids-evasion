"""Training regimes.

baseline        standard training on clean data.
fgsm_static_at  "static" FGSM adversarial training: FGSM examples are crafted
                ONCE against a pre-trained baseline and added to the training
                set. Cheap and common in course projects, and it fails: the
                model learns to resist a fixed set of perturbations and flattens
                its gradients (gradient masking), so FGSM looks defeated while
                an iterative attack (PGD) still succeeds.
pgd_at          Madry-style adversarial training: PGD examples are regenerated
                every mini-batch against the *current* model (50% clean,
                50% adversarial per batch).
"""
from __future__ import annotations

import numpy as np

from .attacks import fgsm, pgd
from .model import MLP


def _batches(n, bs, rng):
    idx = rng.permutation(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def train_baseline(X, y, seed=0, epochs=10, bs=256, lr=1e-3, verbose=False):
    rng = np.random.default_rng(seed)
    m = MLP(X.shape[1], seed=seed)
    for ep in range(epochs):
        losses = [m.step(X[b], y[b], lr) for b in _batches(len(X), bs, rng)]
        if verbose:
            print(f"  [baseline] epoch {ep + 1}/{epochs} loss={np.mean(losses):.4f}")
    return m


def train_fgsm_static_at(X, y, mask, eps_train, seed=0, epochs=10, bs=256,
                         lr=1e-3, verbose=False):
    base = train_baseline(X, y, seed=seed, epochs=epochs, bs=bs, lr=lr)
    atk = y == 1
    X_adv = X.copy()
    X_adv[atk] = fgsm(base, X[atk], y[atk], eps_train, mask)  # crafted once
    X_aug = np.vstack([X, X_adv[atk]])
    y_aug = np.concatenate([y, y[atk]])
    rng = np.random.default_rng(seed + 1000)
    m = MLP(X.shape[1], seed=seed + 1000)
    for ep in range(epochs):
        losses = [m.step(X_aug[b], y_aug[b], lr) for b in _batches(len(X_aug), bs, rng)]
        if verbose:
            print(f"  [fgsm_static_at] epoch {ep + 1}/{epochs} loss={np.mean(losses):.4f}")
    return m


def train_pgd_at(X, y, mask, eps_train, seed=0, epochs=10, bs=256, lr=1e-3,
                 steps=7, verbose=False):
    rng = np.random.default_rng(seed)
    m = MLP(X.shape[1], seed=seed)
    for ep in range(epochs):
        losses = []
        for b in _batches(len(X), bs, rng):
            Xb, yb = X[b], y[b]
            atk = yb == 1
            Xa = Xb.copy()
            if atk.any():
                Xa[atk] = pgd(m, Xb[atk], yb[atk], eps_train, mask, steps=steps, rng=rng)
            half = rng.random(len(Xb)) < 0.5  # 50% clean / 50% adversarial
            Xmix = np.where(half[:, None], Xa, Xb)
            losses.append(m.step(Xmix, yb, lr))
        if verbose:
            print(f"  [pgd_at] epoch {ep + 1}/{epochs} loss={np.mean(losses):.4f}")
    return m
