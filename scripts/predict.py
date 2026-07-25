#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simulate crop growth with a (fine-tuned) DeepCGM-generic checkpoint.

    python scripts/predict.py --data data/demo_nefer \
        --checkpoint outputs/demo_finetuned --split test \
        --output outputs/demo_predictions

Writes ``predictions.csv`` (daily simulated state in physical units) and, when
the dataset carries observations, ``metrics.json``.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepcgm import CultivarIndex, DeepCGMGeneric, load_dataset  # noqa: E402
from deepcgm.metrics import mean_nrmse, summarise  # noqa: E402
from deepcgm.model import DEFAULT_CHECKPOINT  # noqa: E402
from deepcgm.scaling import unscale_outputs  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--split", default="all",
                   help="value of the 'split' column to simulate; 'all' uses every row")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--plot", action="store_true",
                   help="also write one time-series figure per plot-season (needs matplotlib)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model = DeepCGMGeneric.from_pretrained(args.checkpoint, device=args.device)
    index = CultivarIndex.from_checkpoint(args.checkpoint / "cultivars.json",
                                          num_slots=model.config.num_cultivar_slots)

    splits = None if args.split == "all" else [args.split]
    batch = load_dataset(args.data, model.input_scaler, model.config.output_scale,
                         index, splits=splits, allow_new_cultivars=False)

    features = list(model.config.output_features)
    prediction = model.predict(batch.drivers, batch.cultivar_ids, batch_size=args.batch_size)
    observation = unscale_outputs(batch.targets.numpy(), model.config.output_scale)

    args.output.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, uid in enumerate(batch.uids):
        frame = pd.DataFrame(prediction[i], columns=features)
        frame.insert(0, "das", np.arange(prediction.shape[1]))
        frame.insert(0, "date", batch.dates[i])
        frame.insert(0, "cultivar", batch.cultivars[i])
        frame.insert(0, "uid", uid)
        frames.append(frame)
    pd.concat(frames).to_csv(args.output / "predictions.csv", index=False)

    n_obs = int(np.sum(~np.isnan(observation)))
    if n_obs:
        report = {
            "n_plot_seasons": len(batch),
            "n_observations": n_obs,
            "overall": summarise(prediction, observation, features),
            "mean_nRMSE": mean_nrmse(prediction, observation, features),
            "per_cultivar": {},
        }
        for cultivar in dict.fromkeys(batch.cultivars):
            mask = [c == cultivar for c in batch.cultivars]
            report["per_cultivar"][cultivar] = {
                "n_plot_seasons": int(sum(mask)),
                "mean_nRMSE": mean_nrmse(prediction[mask], observation[mask], features),
                "variables": summarise(prediction[mask], observation[mask], features),
            }
        with open(args.output / "metrics.json", "w") as f:
            json.dump(report, f, indent=2)

        print(f"{len(batch)} plot-seasons, {n_obs} observations")
        print(f"{'variable':>8} {'n':>5} {'R2':>8} {'RMSE':>12} {'nRMSE':>8}")
        for name, m in report["overall"].items():
            print(f"{name:>8} {m['n']:>5} {m['R2']:>8.3f} {m['RMSE']:>12.2f} {m['nRMSE']:>8.3f}")
        print(f"mean nRMSE over LAI/TWLV/TWST/WSO/TAGP: {report['mean_nRMSE']:.3f}")
    else:
        print(f"{len(batch)} plot-seasons simulated (no observations to score against)")

    if args.plot:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from plot_timeseries import plot_batch  # noqa: E402
        plot_batch(batch, prediction, observation, features, args.output / "figures")

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
