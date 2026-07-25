#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Time-series figures: simulated curves with the observations on top.

Used by ``scripts/predict.py --plot``, and callable on its own::

    python scripts/plot_timeseries.py --data data/demo_nefer \
        --checkpoint outputs/demo_finetuned --split test --output outputs/figures
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

UNITS = {"DVS": "-", "LAI": "m2/m2", "TWLV": "kg/ha", "TWST": "kg/ha",
         "WSO": "kg/ha", "TAGP": "kg/ha"}


def plot_batch(batch, prediction, observation, features, output_dir, dpi=150):
    """One figure per plot-season, one panel per output variable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    das = np.arange(prediction.shape[1])

    for i, uid in enumerate(batch.uids):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
        for j, (name, ax) in enumerate(zip(features, axes.ravel())):
            ax.plot(das, prediction[i, :, j], color="#c44e52", lw=1.6,
                    label="DeepCGM-generic")
            obs = observation[i, :, j]
            seen = ~np.isnan(obs)
            if seen.any():
                ax.plot(das[seen], obs[seen], "o", ms=4, color="#333333",
                        label="observed")
            label = f"{name} ({UNITS.get(name, '-')})"
            if name == "DVS":
                label += "  [input]"   # phenology is driven, not simulated
            ax.set_ylabel(label)
            ax.grid(alpha=0.3)
            if j >= 3:
                ax.set_xlabel("days after sowing")
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{uid} - {batch.cultivars[i]}", fontsize=11)
        fig.tight_layout()
        fig.savefig(output_dir / f"{uid}.png", dpi=dpi)
        plt.close(fig)

    return output_dir


def main(argv=None):
    from deepcgm import CultivarIndex, DeepCGMGeneric, load_dataset
    from deepcgm.model import DEFAULT_CHECKPOINT
    from deepcgm.scaling import unscale_outputs

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--split", default="all")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    model = DeepCGMGeneric.from_pretrained(args.checkpoint, device=args.device)
    index = CultivarIndex.from_checkpoint(args.checkpoint / "cultivars.json",
                                          num_slots=model.config.num_cultivar_slots)
    splits = None if args.split == "all" else [args.split]
    batch = load_dataset(args.data, model.input_scaler, model.config.output_scale,
                         index, splits=splits, allow_new_cultivars=False)

    prediction = model.predict(batch.drivers, batch.cultivar_ids)
    observation = unscale_outputs(batch.targets.numpy(), model.config.output_scale)
    out = plot_batch(batch, prediction, observation,
                     list(model.config.output_features), args.output)
    print(f"wrote {len(batch)} figures to {out}")


if __name__ == "__main__":
    main()
