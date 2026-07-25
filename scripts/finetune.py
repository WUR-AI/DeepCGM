#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune DeepCGM-generic on new cultivars.

Only the cultivar embeddings are optimised; the backbone and the hypernetwork
stay frozen, so adapting to a cultivar costs 32 trainable parameters. This is
the "calibration" stage of the paper.

    python scripts/finetune.py --data data/demo_nefer --output outputs/demo_finetuned

The output directory is itself a checkpoint: pass it to ``scripts/predict.py``.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepcgm import CultivarIndex, DeepCGMGeneric, load_dataset  # noqa: E402
from deepcgm.losses import reference_denominators, total_loss  # noqa: E402
from deepcgm.model import DEFAULT_CHECKPOINT  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, type=Path,
                   help="dataset directory with plots.csv / drivers.csv / observations.csv")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help="pretrained checkpoint to start from")
    p.add_argument("--output", required=True, type=Path,
                   help="where to write the fine-tuned checkpoint")
    p.add_argument("--split", default="finetune",
                   help="value of the 'split' column to fine-tune on; "
                        "use 'all' to ignore the column")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lr-decay", type=float, default=0.8,
                   help="multiplicative decay applied every --lr-decay-every epochs")
    p.add_argument("--lr-decay-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-convergence-loss", action="store_true",
                   help="drop the convergence term (faster, slightly less stable)")
    p.add_argument("--warm-start", action="store_true",
                   help="keep the pretrained vector for cultivars the model already "
                        "knows instead of restarting them from zero")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--reproduce-reference-batch", action="store_true",
                   help="normalise the loss with the observation counts of the batch the "
                        "checkpoint was originally fine-tuned on, instead of this batch's "
                        "own counts. Use it to reproduce a released cultivar embedding from "
                        "that cultivar's data alone; leave it off for your own data.")
    p.add_argument("--num-cultivar-slots", type=int, default=None,
                   help="enlarge the embedding table; needed only when fine-tuning "
                        "on more cultivars than it has free rows (200 by default, "
                        "62 of them already taken)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = DeepCGMGeneric.from_pretrained(args.checkpoint, device=args.device,
                                           num_cultivar_slots=args.num_cultivar_slots)
    index = CultivarIndex.from_checkpoint(args.checkpoint / "cultivars.json",
                                          num_slots=model.config.num_cultivar_slots)

    splits = None if args.split == "all" else [args.split]
    batch = load_dataset(args.data, model.input_scaler, model.config.output_scale,
                         index, splits=splits).to(args.device)

    trained_ids = sorted(set(batch.cultivar_ids.tolist()))
    if not args.warm_start:
        model.reset_embeddings(trained_ids)

    known = [c for c in dict.fromkeys(batch.cultivars) if index.is_pretrained(c)]
    fresh = [c for c in dict.fromkeys(batch.cultivars) if not index.is_pretrained(c)]
    print(f"{len(batch)} plot-seasons | {len(trained_ids)} cultivars "
          f"({len(fresh)} new, {len(known)} already in the checkpoint)")
    if fresh:
        print("  new cultivars:", ", ".join(fresh))
    print(f"trainable parameters: {len(trained_ids) * model.config.embedding_dim}")

    denominators = None
    if args.reproduce_reference_batch:
        denominators = reference_denominators(model.config)
        if denominators is None:
            raise SystemExit(
                f"{args.checkpoint}/config.json has no 'reference_loss_counts', so the "
                "original batch cannot be reproduced; drop --reproduce-reference-batch")
        print("normalising the loss with the reference batch counts "
              f"{[int(c) for c in denominators['fitting']]} "
              f"(convergence {int(denominators['convergence'])})")

    model.freeze_backbone()
    optimiser = torch.optim.Adam([model.embeddings.embeddings], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda=lambda e: args.lr_decay ** (e // args.lr_decay_every))

    best = {"loss": float("inf"), "epoch": -1,
            "embeddings": model.embeddings.embeddings.detach().clone()}
    history = []
    started = time.time()

    for epoch in range(args.epochs):
        model.train()
        optimiser.zero_grad()
        prediction, (carbon, carbon_sub) = model(batch.drivers, batch.cultivar_ids,
                                                 return_aux=True)
        loss, fit, cg = total_loss(prediction, batch.targets, carbon, carbon_sub,
                                   weights=model.config.loss_weights,
                                   use_convergence=not args.no_convergence_loss,
                                   denominators=denominators)
        loss.backward()
        optimiser.step()
        scheduler.step()

        fit_value, cg_value = fit.detach().item(), cg.detach().item()
        history.append({"epoch": epoch, "loss": fit_value, "convergence_loss": cg_value})
        if fit_value < best["loss"]:
            best = {"loss": fit_value, "epoch": epoch,
                    "embeddings": model.embeddings.embeddings.detach().clone()}

        if args.log_every and (epoch % args.log_every == 0 or epoch == args.epochs - 1):
            eta = (time.time() - started) / (epoch + 1) * (args.epochs - epoch - 1)
            print(f"epoch {epoch:4d} | fitting {fit_value:.5f} | "
                  f"convergence {cg_value:.5f} | lr {scheduler.get_last_lr()[0]:.5f} | "
                  f"eta {eta / 60:.1f} min")

    print(f"best fine-tuning loss {best['loss']:.5f} at epoch {best['epoch']}")

    # --------------------------------------------------------- save checkpoint
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ["backbone.pt", "hypernetwork.pt", "input_scaler.json"]:
        shutil.copy(args.checkpoint / name, args.output / name)

    torch.save({"embeddings": best["embeddings"].cpu()},
               args.output / "cultivar_embeddings.pt")
    with open(args.output / "cultivars.json", "w") as f:
        json.dump({str(v): k for k, v in index.name_to_id.items()}, f,
                  indent=2, sort_keys=True)

    config = json.loads((args.checkpoint / "config.json").read_text())
    config.update({
        "name": config.get("name", "deepcgm-generic") + "-finetuned",
        "num_cultivar_slots": int(best["embeddings"].shape[0]),
        "finetuned_from": str(args.checkpoint),
        "finetuned_on": str(args.data),
        "finetuning_loss": best["loss"],
        "finetuning_epochs": args.epochs,
        "reference_batch_normalisation": bool(args.reproduce_reference_batch),
        "finetuned_cultivars": sorted(dict.fromkeys(batch.cultivars)),
    })
    with open(args.output / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(args.output / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
