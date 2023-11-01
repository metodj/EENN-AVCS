import numpy as np
import random
from typing import List, Tuple


def raps_eenn(
    probs: np.ndarray,
    targets: np.ndarray,
    calib_size: float = 0.2,
    alpha: float = 0.05,
    lam_reg: float = 0.01,
    k_reg: float = 5,
    disallow_zero_sets: bool = False,
    rand: bool = True,
    seed: int = 0,
) -> Tuple[List, List]:
    """
    Code adapted from:
        https://github.com/aangelopoulos/conformal-prediction/blob/main/notebooks/imagenet-raps.ipynb
    """
    L, N, C = probs.shape

    random.seed(seed)
    calib_ids = random.sample(range(N), int(calib_size * N))
    valid_ids = list(set(range(N)) - set(calib_ids))

    reg_vec = np.array(
        k_reg
        * [
            0,
        ]
        + (C - k_reg)
        * [
            lam_reg,
        ]
    )[None, :]

    sizes, coverages, sets, labels = [], [], [], []
    for exit in range(L):
        cal_smx = probs[exit, calib_ids, :]
        cal_labels = targets[calib_ids]
        n = len(cal_labels)

        val_smx = probs[exit, valid_ids, :]
        valid_labels = targets[valid_ids]
        n_valid = len(valid_labels)

        # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
        cal_pi = cal_smx.argsort(1)[:, ::-1]
        cal_srt = np.take_along_axis(cal_smx, cal_pi, axis=1)
        cal_srt_reg = cal_srt + reg_vec
        cal_L = np.where(cal_pi == cal_labels[:, None])[1]
        cal_scores = (
            cal_srt_reg.cumsum(axis=1)[np.arange(n), cal_L]
            - np.random.rand(n) * cal_srt_reg[np.arange(n), cal_L]
        )
        # Get the score quantile
        qhat = np.quantile(
            cal_scores, np.ceil((n + 1) * (1 - alpha)) / n, interpolation="higher"
        )
        # Deploy
        n_val = val_smx.shape[0]
        val_pi = val_smx.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(val_smx, val_pi, axis=1)
        val_srt_reg = val_srt + reg_vec
        indicators = (
            (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val, 1) * val_srt_reg)
            <= qhat
            if rand
            else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
        )
        if disallow_zero_sets:
            indicators[:, 0] = True
        conformal_sets = np.take_along_axis(indicators, val_pi.argsort(axis=1), axis=1)

        sizes.append(conformal_sets.sum(axis=1).mean())
        coverages.append(
            conformal_sets[np.arange(n_valid), valid_labels].sum() / n_valid
        )
        sets.append(conformal_sets)
        labels.append(valid_labels)

    sets = [[list(np.where(x)[0]) for x in sets[l]] for l in range(L)]
    sets = [[sets[l][i] for l in range(L)] for i in range(len(sets[0]))]

    labels = targets[valid_ids]

    return sizes, coverages, sets, labels
