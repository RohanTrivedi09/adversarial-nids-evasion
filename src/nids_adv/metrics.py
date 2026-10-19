"""Security-relevant metrics. Accuracy alone hides the DR/FPR trade-off."""
from __future__ import annotations

import numpy as np


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    dr = tp / max(tp + fn, 1)            # detection rate (recall on attacks)
    fpr = fp / max(fp + tn, 1)           # false alarms on normal traffic
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * dr / max(prec + dr, 1e-12)
    return {"accuracy": (tp + tn) / len(y_true), "detection_rate": dr,
            "fpr": fpr, "precision": prec, "f1": f1,
            "evasion_rate": 1.0 - dr}
