"""White-box evasion attacks (L-infinity), implemented from first principles.

Threat model
------------
* Attacker: white-box access to the model (gradients), controls only the
  malicious flows (label 1). Normal traffic is never perturbed.
* Goal: evasion - make attack traffic be classified as normal.
* Budget: ||delta||_inf <= eps on min-max-scaled features.
* Constraint: only the 38 continuous features may change (`mask`);
  one-hot protocol/service/flag dummies are frozen; features stay in [0, 1].

Because y = 1 for every perturbed sample, *increasing* BCE loss is the same as
pushing the predicted attack probability down, i.e. evasion.
"""
from __future__ import annotations

import numpy as np

from .model import MLP


def _project(X_adv, X, eps, mask):
    delta = np.clip(X_adv - X, -eps, eps) * mask
    return np.clip(X + delta, 0.0, 1.0)


def fgsm(model: MLP, X: np.ndarray, y: np.ndarray, eps: float,
         mask: np.ndarray) -> np.ndarray:
    """Fast Gradient Sign Method (Goodfellow et al., 2015): one step of size eps."""
    if eps == 0:
        return X.copy()
    g = model.input_gradient(X, y)
    return _project(X + eps * np.sign(g), X, eps, mask)


def pgd(model: MLP, X: np.ndarray, y: np.ndarray, eps: float, mask: np.ndarray,
        steps: int = 10, alpha: float | None = None, random_start: bool = True,
        rng: np.random.Generator | None = None) -> np.ndarray:
    """Projected Gradient Descent (Madry et al., 2018)."""
    if eps == 0:
        return X.copy()
    alpha = alpha if alpha is not None else 2.5 * eps / steps
    rng = rng or np.random.default_rng(0)
    X_adv = X.copy()
    if random_start:
        X_adv = _project(X + rng.uniform(-eps, eps, X.shape), X, eps, mask)
    for _ in range(steps):
        g = model.input_gradient(X_adv, y)
        X_adv = _project(X_adv + alpha * np.sign(g), X, eps, mask)
    return X_adv


def perturb_attacks_only(attack_fn, model, X, y, **kw):
    """Apply an attack to malicious rows only (realistic evasion setting)."""
    X_adv = X.copy()
    idx = y == 1
    X_adv[idx] = attack_fn(model, X[idx], y[idx], **kw)
    return X_adv
