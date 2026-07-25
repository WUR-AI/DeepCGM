# -*- coding: utf-8 -*-
"""Input standardisation shipped with the checkpoint.

The pretrained weights only make sense for inputs standardised exactly the way
the pretraining data were standardised, so the fitted mean and scale of the
training set travel with the checkpoint (``input_scaler.json``) instead of
being re-estimated from the user's data.
"""
import json
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class InputScaler:
    """A frozen ``sklearn`` ``StandardScaler`` in plain arrays."""

    features: List[str]
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_json(cls, path) -> "InputScaler":
        with open(path) as f:
            payload = json.load(f)
        return cls(features=list(payload["features"]),
                   mean=np.asarray(payload["mean"], dtype=np.float64),
                   scale=np.asarray(payload["scale"], dtype=np.float64))

    @classmethod
    def identity(cls, features: Sequence[str]) -> "InputScaler":
        """A scaler that leaves the drivers in physical units.

        Only useful while pretraining, to read a dataset once before its own
        mean and scale are known (see ``scripts/pretrain.py``).
        """
        n = len(features)
        return cls(features=list(features), mean=np.zeros(n), scale=np.ones(n))

    @classmethod
    def fit(cls, values: np.ndarray, features: Sequence[str]) -> "InputScaler":
        """Fit on physical-unit drivers whose last axis is ordered like ``features``.

        Constant columns get a scale of 1 rather than 0, which is what
        ``sklearn``'s ``StandardScaler`` does with the wind speed and vapour
        pressure deficit that were held at constants in the paper.
        """
        flat = np.asarray(values, dtype=np.float64).reshape(-1, len(features))
        scale = flat.std(axis=0)
        scale[scale == 0] = 1.0
        return cls(features=list(features), mean=flat.mean(axis=0), scale=scale)

    def to_json(self, path) -> None:
        with open(path, "w") as f:
            json.dump({"features": list(self.features),
                       "mean": [float(v) for v in self.mean],
                       "scale": [float(v) for v in self.scale]}, f, indent=2)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Standardise an array whose last axis is ordered like ``features``."""
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale + self.mean

    def index(self, feature: str) -> int:
        return self.features.index(feature)


def scale_outputs(values: np.ndarray, output_scale: Sequence[float]) -> np.ndarray:
    """Physical units -> the model's internal (roughly unit-range) scale."""
    return np.asarray(values, dtype=np.float64) / np.asarray(output_scale, dtype=np.float64)


def unscale_outputs(values: np.ndarray, output_scale: Sequence[float]) -> np.ndarray:
    """The model's internal scale -> physical units.

    Order is ``[DVS, LAI (m2/m2), TWLV, TWST, WSO, TAGP (kg/ha)]``.
    """
    return np.asarray(values, dtype=np.float64) * np.asarray(output_scale, dtype=np.float64)
