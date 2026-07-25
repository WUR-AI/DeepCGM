#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Two-row summary figure for one cultivar: time series on top, scatter below.

Columns are the five simulated growth variables. The top row shows the daily
trajectory of every plot-season with the observations on top; the bottom row
shows the same observations against the matching simulated value, on a 1:1
plot. WOFOST is drawn alongside DeepCGM-generic when its predictions are
supplied.

    python scripts/plot_cultivar_summary.py --data data/demo_nefer \
        --ensemble checkpoints/ensemble_e32_r2 \
        --wofost data/demo_nefer/wofost_predictions.csv \
        --split test --output outputs/nefer/fig_nefer_test

DeepCGM-generic is the mean over the ensemble members, which is how accuracy is
reported in the paper.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepcgm import CultivarIndex, DeepCGMGeneric, Ensemble, load_dataset  # noqa: E402
from deepcgm.metrics import r2, rmse  # noqa: E402
from deepcgm.model import DEFAULT_CHECKPOINT  # noqa: E402
from deepcgm.scaling import unscale_outputs  # noqa: E402

VARIABLES = ["LAI", "TWLV", "TWST", "WSO", "TAGP"]
UNITS = {"LAI": "m$^2$/m$^2$", "TWLV": "kg/ha", "TWST": "kg/ha",
         "WSO": "kg/ha", "TAGP": "kg/ha"}
COLOURS = {"WOFOST": "#333333", "DeepCGM-generic": "#d95f02"}
OBS_COLOUR = "#1b9e77"
#: vertical position of the statistics block, in axes fractions below the panel
STATS_Y = -0.30


def load_wofost(path, uids, features, n_days):
    """``[n_seasons, n_days, n_features]`` aligned to ``uids``, or None."""
    if path is None:
        return None
    table = pd.read_csv(path)
    missing = [u for u in uids if u not in set(table["uid"])]
    if missing:
        raise SystemExit(f"{path} has no rows for {missing}")
    out = np.full((len(uids), n_days, len(features)), np.nan)
    for i, uid in enumerate(uids):
        g = table[table["uid"] == uid].sort_values("das")
        for j, name in enumerate(features):
            if name in g.columns:
                out[i, :len(g), j] = g[name].to_numpy()[:n_days]
    return out


def plot(batch, predictions, observation, features, output, labels=None, dpi=200):
    """``predictions`` maps a model name to ``[n_seasons, n_days, n_features]``.

    No title is drawn: the figure is meant to be captioned by the document that
    includes it.
    """
    labels = labels or {}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, len(VARIABLES), figsize=(3.0 * len(VARIABLES), 6.6))
    das = np.arange(observation.shape[1])

    for col, name in enumerate(VARIABLES):
        j = features.index(name)
        ax_ts, ax_sc = axes[0, col], axes[1, col]
        obs = observation[:, :, j]
        seen = ~np.isnan(obs)

        # ---------------------------------------------------------- time series
        for model, pred in predictions.items():
            for i in range(pred.shape[0]):
                ax_ts.plot(das, pred[i, :, j], color=COLOURS[model], lw=0.9, alpha=0.75)
        if seen.any():
            ax_ts.scatter(np.tile(das, (obs.shape[0], 1))[seen], obs[seen], s=16,
                          facecolors=OBS_COLOUR, edgecolors="white", linewidths=0.5,
                          zorder=5)
        ax_ts.set_title(name, fontsize=11)
        ax_ts.set_ylabel(f"{name} ({UNITS[name]})" if col == 0 else UNITS[name], fontsize=9)
        ax_ts.set_xlabel("days after planting", fontsize=9)
        ax_ts.grid(alpha=0.25)

        # --------------------------------------------------------------- scatter
        if not seen.any():
            # the variable is still simulated, there is just nothing to score it against
            ax_sc.text(0.5, 0.5, "no observations", ha="center", va="center",
                       transform=ax_sc.transAxes, fontsize=9, color="#777777")
            for spine in ax_sc.spines.values():
                spine.set_edgecolor("#dddddd")
            ax_sc.set_xticks([]); ax_sc.set_yticks([])
            continue

        lines = []
        for model, pred in predictions.items():
            x, y = obs[seen], pred[:, :, j][seen]
            ax_sc.scatter(x, y, s=16, facecolors="none", edgecolors=COLOURS[model],
                          linewidths=0.9)
            lines.append(f"{model}: RMSE {rmse(y, x):.0f}" if name != "LAI"
                         else f"{model}: RMSE {rmse(y, x):.2f}")
            lines[-1] += f", R$^2$ {r2(y, x):.2f}"

        values = [obs[seen]] + [p[:, :, j][seen] for p in predictions.values()]
        lo = min(float(np.nanmin(v)) for v in values)
        hi = max(float(np.nanmax(v)) for v in values)
        pad = 0.05 * (hi - lo if hi > lo else max(abs(hi), 1.0))
        lo, hi = lo - pad, hi + pad
        ax_sc.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color="#808080", alpha=0.8)
        ax_sc.set_xlim(lo, hi); ax_sc.set_ylim(lo, hi)
        ax_sc.set_aspect("equal", adjustable="box")
        ax_sc.set_xlabel(f"observed ({UNITS[name]})", fontsize=9)
        ax_sc.set_ylabel(f"simulated ({UNITS[name]})" if col == 0 else "", fontsize=9)
        ax_sc.grid(alpha=0.25)
        # the statistics go under the panel rather than inside it, so that no
        # point is ever hidden behind them
        ax_sc.text(0.5, STATS_Y, f"n = {int(seen.sum())}\n" + "\n".join(lines),
                   transform=ax_sc.transAxes, va="top", ha="center", fontsize=7,
                   color="#222222")

    handles = [Line2D([0], [0], marker="o", ls="None", color=OBS_COLOUR,
                      markeredgecolor="white", label="observed", markersize=6)]
    handles += [Line2D([0], [0], color=COLOURS[m], lw=1.4, label=labels.get(m, m))
                for m in predictions]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    # leave room under the bottom row for the per-panel statistics
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--ensemble", type=Path,
                   help="directory of seed_* checkpoints; the ensemble mean is plotted")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help="single checkpoint, used when --ensemble is not given")
    p.add_argument("--wofost", type=Path, help="CSV of WOFOST predictions to overlay")
    p.add_argument("--split", default="test")
    p.add_argument("--output", required=True, type=Path,
                   help="output path without extension; .png/.pdf/.svg are written")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    if args.ensemble:
        model = Ensemble.from_directory(args.ensemble, device=args.device)
        cultivars_json = sorted(args.ensemble.glob("seed_*/cultivars.json"))[0]
        label = f"DeepCGM-generic ({len(model)}-seed mean)"
    else:
        model = DeepCGMGeneric.from_pretrained(args.checkpoint, device=args.device)
        cultivars_json = args.checkpoint / "cultivars.json"
        label = "DeepCGM-generic"

    index = CultivarIndex.from_checkpoint(cultivars_json,
                                          num_slots=model.config.num_cultivar_slots)
    splits = None if args.split == "all" else [args.split]
    batch = load_dataset(args.data, model.input_scaler, model.config.output_scale,
                         index, splits=splits, allow_new_cultivars=False)

    features = list(model.config.output_features)
    observation = unscale_outputs(batch.targets.numpy(), model.config.output_scale)
    deep = model.predict(batch.drivers, batch.cultivar_ids)

    predictions = {}
    wofost = load_wofost(args.wofost, batch.uids, features, observation.shape[1])
    if wofost is not None:
        predictions["WOFOST"] = wofost
    predictions["DeepCGM-generic"] = deep

    out = plot(batch, predictions, observation, features, args.output,
               labels={"DeepCGM-generic": label})
    print(f"wrote {out.with_suffix('.png')} (+ .pdf, .svg)")

    print(f"\n{'variable':>8} {'n':>5} " + " ".join(f"{m:>28}" for m in predictions))
    for name in VARIABLES:
        j = features.index(name)
        obs = observation[:, :, j]
        seen = ~np.isnan(obs)
        if not seen.any():
            print(f"{name:>8} {0:>5}")
            continue
        cells = []
        for m, pred in predictions.items():
            y = pred[:, :, j][seen]
            cells.append(f"R2 {r2(y, obs[seen]):>6.3f}  RMSE {rmse(y, obs[seen]):>9.2f}")
        print(f"{name:>8} {int(seen.sum()):>5} " + " ".join(f"{c:>28}" for c in cells))


if __name__ == "__main__":
    main()
