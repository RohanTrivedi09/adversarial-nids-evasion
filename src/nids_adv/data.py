"""NSL-KDD loading and preprocessing.

- 41 raw features: 3 categorical (protocol_type, service, flag) + 38 continuous.
- Categorical -> one-hot (vocabulary fit on the training set only).
- Continuous -> log1p for heavy-tailed byte/count columns, then min-max to [0, 1]
  using training-set statistics; test values are clipped into [0, 1].
- Label: binary, 0 = normal, 1 = attack.

The returned `cont_mask` marks the continuous columns. Attacks only perturb
those columns: flipping a one-hot "service" or "protocol" dummy by 0.1 has no
meaning for real traffic, so it is excluded from the threat model.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass

import numpy as np
import pandas as pd

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
HEAVY_TAILED = ["duration", "src_bytes", "dst_bytes", "hot", "num_compromised",
                "num_root", "num_file_creations", "num_access_files", "count",
                "srv_count", "dst_host_count", "dst_host_srv_count"]

BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = {"train": "KDDTrain+.txt", "test": "KDDTest+.txt"}


@dataclass
class Dataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    cont_mask: np.ndarray  # bool, True = continuous (attackable) feature


def download(data_dir: str = "data") -> None:
    os.makedirs(data_dir, exist_ok=True)
    for fname in FILES.values():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"downloading {fname} ...")
            urllib.request.urlretrieve(BASE_URL + fname, path)


def _read(path: str) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=COLUMNS)


def load_nsl_kdd(data_dir: str = "data") -> Dataset:
    download(data_dir)
    tr = _read(os.path.join(data_dir, FILES["train"]))
    te = _read(os.path.join(data_dir, FILES["test"]))

    y_tr = (tr["label"] != "normal").astype(np.float64).to_numpy()
    y_te = (te["label"] != "normal").astype(np.float64).to_numpy()

    cont_cols = [c for c in COLUMNS[:-2] if c not in CATEGORICAL]
    assert len(cont_cols) == 38

    def cont_block(df: pd.DataFrame) -> pd.DataFrame:
        out = df[cont_cols].astype(np.float64).copy()
        for c in HEAVY_TAILED:
            out[c] = np.log1p(out[c])
        return out

    ctr, cte = cont_block(tr), cont_block(te)
    lo, hi = ctr.min(), ctr.max()
    span = (hi - lo).replace(0, 1.0)
    ctr = (ctr - lo) / span
    cte = ((cte - lo) / span).clip(0.0, 1.0)

    # one-hot, vocabulary from the training set only
    cat_tr = pd.get_dummies(tr[CATEGORICAL].astype(str), dtype=np.float64)
    cat_te = pd.get_dummies(te[CATEGORICAL].astype(str), dtype=np.float64)
    cat_te = cat_te.reindex(columns=cat_tr.columns, fill_value=0.0)

    X_tr = np.hstack([ctr.to_numpy(), cat_tr.to_numpy()])
    X_te = np.hstack([cte.to_numpy(), cat_te.to_numpy()])
    names = cont_cols + list(cat_tr.columns)
    mask = np.zeros(X_tr.shape[1], dtype=bool)
    mask[: len(cont_cols)] = True
    return Dataset(X_tr, y_tr, X_te, y_te, names, mask)
