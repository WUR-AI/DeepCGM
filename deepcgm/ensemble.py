# -*- coding: utf-8 -*-
"""Running several independently pretrained models as one ensemble.

The accuracy reported in the paper is that of the **ensemble mean over 10
random seeds**: the ten models' daily predictions are averaged and the metrics
are computed once on that average. A single seed is measurably worse, so
reproducing a published number requires all ten.

    from deepcgm import Ensemble
    ensemble = Ensemble.from_directory("checkpoints/ensemble_e32_r2")
    mean = ensemble.predict(batch.drivers, batch.cultivar_ids)          # [N, 365, 6]
    per_seed = ensemble.predict(batch.drivers, batch.cultivar_ids, reduce=None)
"""
import json
import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from .model import DeepCGMGeneric


def _seed_of(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else 0


class Ensemble:
    """A set of checkpoints that are averaged in prediction space."""

    def __init__(self, models: Sequence[DeepCGMGeneric], seeds: Optional[Sequence[int]] = None):
        if not models:
            raise ValueError("an ensemble needs at least one model")
        self.models: List[DeepCGMGeneric] = list(models)
        self.seeds = list(seeds) if seeds is not None else list(range(len(self.models)))
        self.config = self.models[0].config
        self.input_scaler = self.models[0].input_scaler

    def __len__(self) -> int:
        return len(self.models)

    @classmethod
    def from_directory(cls, directory, device: str = "cpu") -> "Ensemble":
        """Load every ``seed_*`` subdirectory of ``directory``."""
        directory = Path(directory)
        members = sorted((p for p in directory.iterdir()
                          if p.is_dir() and p.name.startswith("seed_")), key=_seed_of)
        if not members:
            raise ValueError(f"no seed_* subdirectories found in {directory}")
        models = [DeepCGMGeneric.from_pretrained(p, device=device) for p in members]
        return cls(models, [_seed_of(p) for p in members])

    @property
    def manifest(self) -> Optional[dict]:
        return getattr(self, "_manifest", None)

    def predict(self, drivers: torch.Tensor, cultivar_ids: torch.Tensor,
                reduce: Optional[str] = "mean", batch_size: int = 64) -> np.ndarray:
        """Simulate with every member.

        ``reduce="mean"`` returns the ensemble mean ``[N, 365, 6]``, which is
        what the paper reports; ``reduce=None`` returns every member's
        prediction ``[n_seeds, N, 365, 6]``, useful for uncertainty bands.
        """
        stacked = np.stack([m.predict(drivers, cultivar_ids, batch_size=batch_size)
                            for m in self.models])
        if reduce is None:
            return stacked
        if reduce == "mean":
            return stacked.mean(axis=0)
        if reduce == "median":
            return np.median(stacked, axis=0)
        raise ValueError(f"unknown reduce={reduce!r}")
