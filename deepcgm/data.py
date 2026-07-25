# -*- coding: utf-8 -*-
"""Reading plot-season data in the repository's CSV format.

A dataset is a directory with three files::

    plots.csv         one row per plot-season (uid, cultivar, planting_date, split, ...)
    drivers.csv       one row per plot-season and day (uid, date, IRRAD, ..., DVS)
    observations.csv  sparse measurements (uid, date, DVS, LAI, TWLV, TWST, WSO, TAGP)

See ``docs/data_format.md`` for the column definitions and units.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .scaling import InputScaler, scale_outputs

SEQUENCE_LENGTH = 365
WEATHER_FEATURES = ("IRRAD", "TMIN", "TMAX", "RAIN")
MANAGEMENT_FEATURES = ("irr", "fer")


@dataclass
class SeasonBatch:
    """A batch of plot-seasons, ready to be fed to the model."""

    uids: List[str]
    cultivars: List[str]
    cultivar_ids: torch.Tensor          # [N]           long, index into the embedding table
    drivers: torch.Tensor               # [N, 365, 7]   standardised
    targets: torch.Tensor               # [N, 365, 6]   scaled, NaN where unobserved
    dates: np.ndarray                   # [N, 365]      datetime64[D]

    def __len__(self) -> int:
        return len(self.uids)

    def to(self, device) -> "SeasonBatch":
        return SeasonBatch(self.uids, self.cultivars, self.cultivar_ids.to(device),
                           self.drivers.to(device), self.targets.to(device), self.dates)

    def subset(self, mask: Sequence[bool]) -> "SeasonBatch":
        idx = [i for i, keep in enumerate(mask) if keep]
        return SeasonBatch([self.uids[i] for i in idx],
                           [self.cultivars[i] for i in idx],
                           self.cultivar_ids[idx], self.drivers[idx],
                           self.targets[idx], self.dates[idx])


class CultivarIndex:
    """Maps cultivar names onto rows of the embedding table.

    Names already present in the released checkpoint keep their original row,
    so a pretrained embedding is reused. Unseen names are assigned to free rows
    and start from a zero vector, exactly as the held-out cultivars did in the
    paper.
    """

    def __init__(self, name_to_id: Optional[Dict[str, int]] = None, num_slots: int = 200):
        self.num_slots = num_slots
        self.name_to_id: Dict[str, int] = dict(name_to_id or {})
        self.new_names: List[str] = []

    @classmethod
    def from_checkpoint(cls, cultivars_json: Path, num_slots: int = 200) -> "CultivarIndex":
        import json
        with open(cultivars_json) as f:
            id_to_name = json.load(f)
        return cls({name: int(code) for code, name in id_to_name.items()}, num_slots)

    def _free_slot(self) -> int:
        used = set(self.name_to_id.values())
        for slot in range(self.num_slots):
            if slot not in used:
                return slot
        raise ValueError(
            f"the embedding table only has {self.num_slots} rows and all are taken; "
            "enlarge it with scripts/finetune.py --num-cultivar-slots")

    def get(self, name: str, allow_new: bool = True) -> int:
        if name in self.name_to_id:
            return self.name_to_id[name]
        if not allow_new:
            raise KeyError(f"unknown cultivar {name!r}; it has no fitted embedding")
        slot = self._free_slot()
        self.name_to_id[name] = slot
        self.new_names.append(name)
        return slot

    def is_pretrained(self, name: str) -> bool:
        return name in self.name_to_id and name not in self.new_names


def _align_season(group: pd.DataFrame, planting: pd.Timestamp,
                  features: Sequence[str]) -> pd.DataFrame:
    """Put one plot-season on a fixed 365-day calendar starting at planting."""
    calendar = pd.date_range(planting, periods=SEQUENCE_LENGTH, freq="D")
    aligned = group.set_index("date").reindex(calendar)

    for col in WEATHER_FEATURES:
        if col in aligned:
            aligned[col] = aligned[col].ffill().bfill()
    for col in MANAGEMENT_FEATURES:
        if col in aligned:
            aligned[col] = aligned[col].fillna(0.0)
    if "DVS" in aligned:
        # DVS is monotonic; carry the last known value forward, and use the
        # first known value before emergence
        aligned["DVS"] = aligned["DVS"].ffill().bfill()

    missing = [c for c in features if aligned[c].isna().any()]
    if missing:
        raise ValueError(
            f"plot-season starting {planting.date()} still has gaps in {missing} after "
            "forward/backward filling; check drivers.csv")
    aligned.index.name = "date"
    return aligned


def load_dataset(directory, scaler: InputScaler, output_scale: Sequence[float],
                 cultivar_index: CultivarIndex, splits: Optional[Iterable[str]] = None,
                 allow_new_cultivars: bool = True,
                 output_features: Sequence[str] = ("DVS", "LAI", "TWLV", "TWST", "WSO", "TAGP"),
                 ) -> SeasonBatch:
    """Read a dataset directory into a :class:`SeasonBatch`."""
    directory = Path(directory)
    plots = pd.read_csv(directory / "plots.csv", parse_dates=["planting_date"])
    drivers = pd.read_csv(directory / "drivers.csv", parse_dates=["date"])
    obs_path = directory / "observations.csv"
    observations = (pd.read_csv(obs_path, parse_dates=["date"])
                    if obs_path.exists() else pd.DataFrame(columns=["uid", "date"]))

    if splits is not None:
        wanted = set(splits)
        if "split" not in plots.columns:
            raise ValueError("plots.csv has no 'split' column but splits were requested")
        plots = plots[plots["split"].isin(wanted)]
    if plots.empty:
        raise ValueError(f"no plot-seasons selected in {directory}")

    features = list(scaler.features)
    absent = [c for c in features if c not in drivers.columns]
    if absent:
        hint = (" — DVS is a required input; supply it in drivers.csv, e.g. from a "
                "calibrated WOFOST/PCSE run") if absent == ["DVS"] else ""
        raise ValueError(f"drivers.csv is missing the column(s) {absent}{hint}")

    driver_groups = {uid: g for uid, g in drivers.groupby("uid")}
    obs_groups = {uid: g for uid, g in observations.groupby("uid")}

    X, Y, dates, ids, uids, cultivars = [], [], [], [], [], []
    for row in plots.itertuples(index=False):
        uid = row.uid
        if uid not in driver_groups:
            raise ValueError(f"no rows in drivers.csv for plot-season {uid!r}")
        aligned = _align_season(driver_groups[uid], row.planting_date, features)
        X.append(scaler.transform(aligned[features].to_numpy(dtype=np.float64)))

        target = np.full((SEQUENCE_LENGTH, len(output_features)), np.nan)
        if uid in obs_groups:
            obs = obs_groups[uid].set_index("date").reindex(aligned.index)
            present = [c for c in output_features if c in obs.columns]
            target[:, [output_features.index(c) for c in present]] = \
                obs[present].to_numpy(dtype=np.float64)
        Y.append(scale_outputs(target, output_scale))

        dates.append(aligned.index.to_numpy(dtype="datetime64[D]"))
        cultivars.append(row.cultivar)
        ids.append(cultivar_index.get(row.cultivar, allow_new=allow_new_cultivars))
        uids.append(uid)

    return SeasonBatch(
        uids=uids,
        cultivars=cultivars,
        cultivar_ids=torch.tensor(ids, dtype=torch.long),
        drivers=torch.tensor(np.stack(X), dtype=torch.float32),
        targets=torch.tensor(np.stack(Y), dtype=torch.float32),
        dates=np.stack(dates),
    )
