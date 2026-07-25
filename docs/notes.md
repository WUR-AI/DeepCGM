# Practical notes

The [README](../README.md) has everything needed to run the model. This file
covers the parts that can surprise you once you move past the demo and onto
your own data.

- [DVS is an input, not an output](#dvs-is-an-input-not-an-output)
- [Observations may be as sparse as you like](#observations-may-be-as-sparse-as-you-like)
- [Cultivar names are matched against the checkpoint](#cultivar-names-are-matched-against-the-checkpoint)
- [Batch composition changes the result](#batch-composition-changes-the-result)
- [What pretraining does differently](#what-pretraining-does-differently)

---

## DVS is an input you must provide

DeepCGM does not simulate phenology; it is told the development stage each day
and simulates biomass and LAI given that timing. The DVS column is therefore a
required input that you supply yourself — the model does not synthesise it. The
paper obtained DVS from a calibrated WOFOST run; if you have
[PCSE](https://github.com/ajwdewit/pcse), that is the recommended route. The
quality of your DVS caps the quality of everything downstream.

## Observations may be as sparse as you like

The loss is masked, so a cultivar with only aboveground biomass measurements
contributes only through that variable. Fine-tuning still works, but the paper
found that cultivars with few fine-tuning observations end up with less
reliable predictions in the variables they never constrained.

## Cultivar names are matched against the checkpoint

A name that already appears in `cultivars.json` reuses its fitted embedding;
anything else is treated as a new cultivar and starts from a zero vector. Pass
`--warm-start` to `finetune.py` to keep the existing vector as the starting
point for known cultivars.

## Batch composition changes the result

Worth knowing before you fine-tune several cultivars at once: the loss divides
each variable's error by the number of observations of that variable **in the
batch**. Fine-tuning cultivars together therefore does not give the same
embeddings as fine-tuning them one at a time — the denominators differ, and
they differ by a different factor per variable, which tilts the gradient
towards whichever variables are scarce in that particular batch. Neither
answer is wrong, but they are not interchangeable, so keep your batches
consistent if you compare runs.

The checkpoints record the counts of the batch they were originally fitted on
(`reference_loss_counts` in `config.json`). Passing
`--reproduce-reference-batch` reuses them, which makes a single-cultivar run
follow the original trajectory — useful if you want to re-derive one of the
shipped embeddings from that cultivar's data alone:

| fine-tuning of Nefer's 13 plot-seasons | cosine to the shipped embedding |
| --- | --- |
| default (this batch's own counts) | 0.821 |
| `--reproduce-reference-batch` | 0.99999982 |

Leave the flag off for your own data: there is no original batch to match, and
your batch's own counts are the right normalisation.

## What pretraining does differently

Two differences from fine-tuning are worth knowing. Pretraining **fits its own
input scaler** (there is no checkpoint to inherit one from) and writes it to the
output directory, because every later stage must standardise its drivers exactly
the same way. And it uses the paper's differential learning rates — 0.1 for the
backbone, 0.01 for the embeddings and hypernetwork — so the cultivar-specific
part does not race ahead of the mechanism it modulates.

To check the architecture, parameter counts and optimiser setup without any
data at all:

```bash
python scripts/pretrain.py --dry-run
```

```
backbone             20,802 parameters  (lr 0.1)
hypernetwork         44,880 parameters  (lr 0.01)
embedding table       6,400 parameters  (lr 0.01), 200 x 32
```

The rest of the training configuration — optimiser, epochs, loss weights, the
absence of a validation split — is in [model_card.md](model_card.md).
