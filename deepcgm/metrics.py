# -*- coding: utf-8 -*-
"""Evaluation metrics, in physical units."""
from typing import Dict, Sequence

import numpy as np


def _paired(prediction: np.ndarray, observation: np.ndarray):
    mask = ~np.isnan(observation)
    return prediction[mask], observation[mask]


def rmse(prediction: np.ndarray, observation: np.ndarray) -> float:
    p, o = _paired(prediction, observation)
    return float("nan") if p.size == 0 else float(np.sqrt(np.mean((p - o) ** 2)))


def r2(prediction: np.ndarray, observation: np.ndarray) -> float:
    p, o = _paired(prediction, observation)
    if p.size < 2:
        return float("nan")
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - np.mean(o)) ** 2)
    return float("nan") if ss_tot == 0 else float(1 - ss_res / ss_tot)


def nrmse(prediction: np.ndarray, observation: np.ndarray, min_observations: int = 2) -> float:
    """RMSE divided by the absolute mean of the observations.

    Returns NaN when fewer than ``min_observations`` points are available, which
    is the convention used for the cultivar-level comparison in the paper.
    """
    p, o = _paired(prediction, observation)
    if p.size < min_observations or np.mean(o) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - o) ** 2)) / abs(np.mean(o)))


def summarise(prediction: np.ndarray, observation: np.ndarray,
              features: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Per-variable R2, RMSE, nRMSE and observation count.

    Both arrays are ``[n_seasons, n_days, n_features]`` in physical units.
    """
    out = {}
    for i, name in enumerate(features):
        p, o = prediction[..., i], observation[..., i]
        out[name] = {
            "n": int(np.sum(~np.isnan(o))),
            "R2": r2(p, o),
            "RMSE": rmse(p, o),
            "nRMSE": nrmse(p, o),
        }
    return out


def mean_nrmse(prediction: np.ndarray, observation: np.ndarray,
               features: Sequence[str],
               variables: Sequence[str] = ("LAI", "TWLV", "TWST", "WSO", "TAGP")) -> float:
    """The cultivar-level nRMSE of the paper: the mean over the variables that
    have at least two observations."""
    values = [nrmse(prediction[..., features.index(v)], observation[..., features.index(v)])
              for v in variables if v in features]
    values = [v for v in values if not np.isnan(v)]
    return float("nan") if not values else float(np.mean(values))
