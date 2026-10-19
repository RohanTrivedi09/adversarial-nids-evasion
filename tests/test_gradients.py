"""Sanity checks: the hand-written gradients and attack constraints are correct."""
import numpy as np

from nids_adv.attacks import fgsm, pgd
from nids_adv.model import MLP


def _toy(n=8, d=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.1, 0.9, size=(n, d))
    y = rng.integers(0, 2, size=n).astype(float)
    mask = np.zeros(d, dtype=bool)
    mask[:8] = True
    return X, y, mask


def test_input_gradient_matches_finite_differences():
    X, y, _ = _toy()
    m = MLP(X.shape[1], seed=1)
    g = m.input_gradient(X, y)
    h = 1e-6
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Xp, Xm = X.copy(), X.copy()
            Xp[i, j] += h
            Xm[i, j] -= h
            lp = m.loss(m.predict_proba(Xp[i:i + 1]), y[i:i + 1])
            lm = m.loss(m.predict_proba(Xm[i:i + 1]), y[i:i + 1])
            assert abs((lp - lm) / (2 * h) - g[i, j]) < 1e-5


def test_attacks_respect_budget_mask_and_box():
    X, y, mask = _toy(n=32)
    y[:] = 1.0
    m = MLP(X.shape[1], seed=2)
    for eps in (0.05, 0.2):
        for X_adv in (fgsm(m, X, y, eps, mask), pgd(m, X, y, eps, mask, steps=5)):
            d = X_adv - X
            assert np.all(np.abs(d) <= eps + 1e-12)
            assert np.all(d[:, ~mask] == 0)
            assert X_adv.min() >= 0 and X_adv.max() <= 1


def test_pgd_at_least_as_strong_as_fgsm_on_average():
    X, y, mask = _toy(n=64, seed=3)
    y[:] = 1.0
    m = MLP(X.shape[1], seed=3)
    lf = m.loss(m.predict_proba(fgsm(m, X, y, 0.1, mask)), y)
    lp = m.loss(m.predict_proba(pgd(m, X, y, 0.1, mask, steps=20)), y)
    assert lp >= lf - 1e-6
