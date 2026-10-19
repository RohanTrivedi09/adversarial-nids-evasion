"""A small MLP written from scratch in NumPy.

Architecture: d -> 64 -> 32 -> 1, ReLU hidden units, sigmoid output,
binary cross-entropy loss, Adam optimiser.

`input_gradient` returns dL/dx, which is all FGSM and PGD need. It is checked
against finite differences in tests/test_gradients.py.
"""
from __future__ import annotations

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


class MLP:
    def __init__(self, n_in: int, hidden=(64, 32), seed: int = 0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, 1]
        self.W, self.b = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.W.append(rng.normal(0, np.sqrt(2.0 / a), size=(a, b)))  # He init
            self.b.append(np.zeros(b))
        self._adam_state = None

    # ---------- forward / backward ----------
    def _forward(self, X):
        acts, pre = [X], []
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            pre.append(z)
            h = np.maximum(z, 0) if i < len(self.W) - 1 else _sigmoid(z)
            acts.append(h)
        return acts, pre

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[0][-1].ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.float64)

    @staticmethod
    def loss(p: np.ndarray, y: np.ndarray) -> float:
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def _backward(self, acts, pre, y):
        """Returns (param grads, dL/dx) for mean BCE over the batch."""
        n = y.shape[0]
        delta = (acts[-1].ravel() - y).reshape(-1, 1) / n  # dL/dz for sigmoid+BCE
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            delta = delta @ self.W[i].T
            if i > 0:
                delta = delta * (pre[i - 1] > 0)
        return gW, gb, delta  # final delta is dL/dx

    def input_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gradient of the *per-sample* loss w.r.t. the input (no 1/n factor)."""
        acts, pre = self._forward(X)
        _, _, gx = self._backward(acts, pre, y)
        return gx * X.shape[0]

    # ---------- training ----------
    def step(self, X, y, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        acts, pre = self._forward(X)
        gW, gb, _ = self._backward(acts, pre, y)
        params, grads = self.W + self.b, gW + gb
        if self._adam_state is None:
            self._adam_state = {"t": 0, "m": [np.zeros_like(p) for p in params],
                                "v": [np.zeros_like(p) for p in params]}
        s = self._adam_state
        s["t"] += 1
        for k, (p, g) in enumerate(zip(params, grads)):
            s["m"][k] = b1 * s["m"][k] + (1 - b1) * g
            s["v"][k] = b2 * s["v"][k] + (1 - b2) * g * g
            mh = s["m"][k] / (1 - b1 ** s["t"])
            vh = s["v"][k] / (1 - b2 ** s["t"])
            p -= lr * mh / (np.sqrt(vh) + eps)
        return self.loss(acts[-1].ravel(), y)
