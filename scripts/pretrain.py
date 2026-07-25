#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pretrain DeepCGM-generic from scratch on a multi-cultivar dataset.

This is the first stage of the paper: the backbone, the hypernetwork and the
cultivar embeddings are **all** trainable, and the model learns generic growth
mechanisms together with a representation of every cultivar in the training
set. ``scripts/finetune.py`` is the second stage, where everything but the
embeddings is frozen.

    python scripts/pretrain.py --data <your dataset> --output checkpoints/my_pretrained

The output directory is a checkpoint in the format the rest of the repository
expects, so it can be passed straight to ``finetune.py`` and ``predict.py``.

WHY YOU CANNOT RUN THIS ON THE PAPER'S DATA
-------------------------------------------
The released checkpoints were pretrained on 258 plot-seasons from 45 wheat
cultivars, assembled from four sources with different licences. Not all of them
allow redistribution, so **the pretraining data are not in this repository and
the paper's pretraining run cannot be reproduced from this release alone**.
This script is here so that the pretraining procedure is inspectable and
re-usable, not because it can be executed out of the box.

The four sources and where to get them are listed in the "Code and Data
Availability" section of the paper and in ``docs/data_format.md``; assembling
them into the CSV layout this script reads is the preprocessing pipeline of
Section 2.1. What ships here instead is the *result* of pretraining — the
weights in ``checkpoints/`` — which is what the fine-tuning stage needs.

Pass ``--dry-run`` to build the model and print the training plan without any
data, which is enough to check the architecture, the parameter counts and the
optimiser setup.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepcgm import CultivarIndex, InputScaler, load_dataset  # noqa: E402
from deepcgm.losses import total_loss  # noqa: E402
from deepcgm.model import DeepCGMGeneric, ModelConfig  # noqa: E402

MISSING_DATA_HINT = """\
{path} does not exist.

The pretraining data of the paper are not distributed with this release: they
come from four sources whose licences do not all permit redistribution. See the
module docstring of this script, and the paper's Code and Data Availability
section, for where to obtain them.

To pretrain on data of your own, point --data at a directory holding
plots.csv / drivers.csv / observations.csv as described in docs/data_format.md.
To inspect the procedure without any data, use --dry-run.\
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path,
                   help="dataset directory with plots.csv / drivers.csv / observations.csv")
    p.add_argument("--output", type=Path,
                   help="where to write the pretrained checkpoint")
    p.add_argument("--split", default="pretrain",
                   help="value of the 'split' column to pretrain on; "
                        "use 'all' to ignore the column")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr-backbone", type=float, default=0.1,
                   help="learning rate of the DeepCGM growth gates")
    p.add_argument("--lr-adapter", type=float, default=0.01,
                   help="learning rate of the cultivar embeddings and the hypernetwork")
    p.add_argument("--lr-decay", type=float, default=0.8,
                   help="multiplicative decay applied every --lr-decay-every epochs")
    p.add_argument("--lr-decay-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--embedding-dim", type=int, default=32)
    p.add_argument("--lora-rank", type=int, default=2)
    p.add_argument("--num-cultivar-slots", type=int, default=200,
                   help="rows of the embedding table; leave room for the cultivars "
                        "that will be added at fine-tuning time")
    p.add_argument("--no-convergence-loss", action="store_true",
                   help="drop the convergence term (faster, slightly less stable)")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--dry-run", action="store_true",
                   help="build the model, report the training plan and exit; needs no data")
    return p.parse_args(argv)


def build_model(args, scaler: InputScaler) -> DeepCGMGeneric:
    config = ModelConfig(embedding_dim=args.embedding_dim,
                         lora_rank=args.lora_rank,
                         num_cultivar_slots=args.num_cultivar_slots)
    return DeepCGMGeneric(config, scaler).to(args.device)


def parameter_groups(model: DeepCGMGeneric, args):
    """The differential learning rates of Section 2.4.

    The backbone carries the growth mechanism and is trained fast; the
    embeddings and the hypernetwork that generates the LoRA update are trained
    an order of magnitude slower, so the cultivar-specific part does not race
    ahead of the mechanism it is supposed to modulate.
    """
    return [
        {"params": list(model.backbone.parameters()), "lr": args.lr_backbone},
        {"params": list(model.hypernetwork.parameters()) + [model.embeddings.embeddings],
         "lr": args.lr_adapter},
    ]


def describe(model: DeepCGMGeneric, args, n_seasons=None, n_cultivars=None) -> None:
    backbone = sum(p.numel() for p in model.backbone.parameters())
    hypernet = sum(p.numel() for p in model.hypernetwork.parameters())
    embeddings = model.embeddings.embeddings.numel()
    print(f"backbone           {backbone:>8,} parameters  (lr {args.lr_backbone})")
    print(f"hypernetwork       {hypernet:>8,} parameters  (lr {args.lr_adapter})")
    print(f"embedding table    {embeddings:>8,} parameters  (lr {args.lr_adapter}), "
          f"{model.config.num_cultivar_slots} x {model.config.embedding_dim}")
    print(f"all three are trainable during pretraining; finetune.py freezes the "
          f"first two and leaves {model.config.embedding_dim} per cultivar")
    if n_seasons is not None:
        print(f"data               {n_seasons} plot-seasons | {n_cultivars} cultivars")
    print(f"schedule           {args.epochs} epochs, full batch, Adam, "
          f"lr x{args.lr_decay} every {args.lr_decay_every} epochs")


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dry_run:
        scaler = InputScaler.identity(ModelConfig.input_features)
        describe(build_model(args, scaler), args)
        print("\ndry run: no data were read and no checkpoint was written")
        return

    if args.data is None or args.output is None:
        raise SystemExit("--data and --output are required unless --dry-run is given")
    if not args.data.exists():
        raise SystemExit(MISSING_DATA_HINT.format(path=args.data))

    # ------------------------------------------------------------ the dataset
    # Pretraining fits its own input scaler, because there is no checkpoint to
    # inherit one from. The drivers are therefore read once in physical units
    # and standardised afterwards; the fitted scaler is saved with the weights,
    # since every later stage must reuse it (see deepcgm/scaling.py).
    features = list(ModelConfig.input_features)
    index = CultivarIndex(num_slots=args.num_cultivar_slots)
    splits = None if args.split == "all" else [args.split]
    batch = load_dataset(args.data, InputScaler.identity(features),
                         ModelConfig.output_scale, index, splits=splits)

    scaler = InputScaler.fit(batch.drivers.numpy(), features)
    mean = torch.tensor(scaler.mean, dtype=batch.drivers.dtype)
    scale = torch.tensor(scaler.scale, dtype=batch.drivers.dtype)
    batch.drivers = (batch.drivers - mean) / scale
    batch = batch.to(args.device)

    model = build_model(args, scaler)
    cultivars = sorted(dict.fromkeys(batch.cultivars))
    describe(model, args, n_seasons=len(batch), n_cultivars=len(cultivars))

    # --------------------------------------------------------------- training
    optimiser = torch.optim.Adam(parameter_groups(model, args))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda=lambda e: args.lr_decay ** (e // args.lr_decay_every))

    best = {"loss": float("inf"), "epoch": -1, "state": None}
    history = []
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        optimiser.zero_grad()
        prediction, (carbon, carbon_sub) = model(batch.drivers, batch.cultivar_ids,
                                                 return_aux=True)
        loss, fit, cg = total_loss(prediction, batch.targets, carbon, carbon_sub,
                                   weights=model.config.loss_weights,
                                   use_convergence=not args.no_convergence_loss)
        loss.backward()
        optimiser.step()
        scheduler.step()

        fit_value, cg_value = fit.detach().item(), cg.detach().item()
        history.append({"epoch": epoch, "loss": fit_value, "convergence_loss": cg_value})
        # the paper keeps the backbone with the lowest pretraining loss; there is
        # no independent validation set to stop on (Section 2.4)
        if fit_value < best["loss"]:
            best = {"loss": fit_value, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}

        if args.log_every and (epoch % args.log_every == 0 or epoch == args.epochs - 1):
            eta = (time.time() - started) / (epoch + 1) * (args.epochs - epoch - 1)
            print(f"epoch {epoch:4d} | fitting {fit_value:.5f} | "
                  f"convergence {cg_value:.5f} | lr {scheduler.get_last_lr()[0]:.5f} | "
                  f"eta {eta / 60:.1f} min")

    print(f"best pretraining loss {best['loss']:.5f} at epoch {best['epoch']}")
    model.load_state_dict(best["state"])

    # --------------------------------------------------------- save checkpoint
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.backbone.state_dict(), args.output / "backbone.pt")
    torch.save(model.hypernetwork.state_dict(), args.output / "hypernetwork.pt")
    torch.save({"embeddings": model.embeddings.embeddings.detach().cpu()},
               args.output / "cultivar_embeddings.pt")
    scaler.to_json(args.output / "input_scaler.json")

    with open(args.output / "cultivars.json", "w") as f:
        json.dump({str(v): k for k, v in index.name_to_id.items()}, f,
                  indent=2, sort_keys=True)

    config = {
        "name": "deepcgm-generic-pretrained",
        "description": f"pretrained on {len(batch)} plot-seasons from {len(cultivars)} cultivars",
        "embedding_dim": args.embedding_dim,
        "hypernetwork_hidden_dim": model.config.hypernetwork_hidden_dim,
        "lora_rank": args.lora_rank,
        "lora_alpha": model.config.lora_alpha,
        "organ_size": list(model.config.organ_size),
        "input_mask": model.config.input_mask,
        "num_cultivar_slots": args.num_cultivar_slots,
        "sequence_length": model.config.sequence_length,
        "input_features": list(model.config.input_features),
        "output_features": list(model.config.output_features),
        "output_scale": list(model.config.output_scale),
        "loss_weights": list(model.config.loss_weights),
        "seed": args.seed,
        "pretrained_on": str(args.data),
        "pretraining_loss": best["loss"],
        "pretraining_epochs": args.epochs,
        "pretrained_cultivars": cultivars,
    }
    with open(args.output / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(args.output / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"wrote {args.output}")
    print(f"next: python scripts/finetune.py --checkpoint {args.output} "
          f"--data <new cultivars> --output <fine-tuned>")


if __name__ == "__main__":
    main()
