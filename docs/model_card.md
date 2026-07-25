# Model card — `deepcgm_generic_e32_r2`

## Overview

DeepCGM-generic is a differentiable crop growth model for wheat with a
cultivar-conditioned parameterisation. Daily weather, management and
development stage go in; leaf area index and the biomass of leaves, stems,
storage organs and the whole shoot come out, for 365 days from sowing.

Cultivar identity enters once, as a 32-dimensional embedding, and is turned by
a hypernetwork into a rank-2 update of the growth-gate weights. Adapting the
model to a new cultivar means optimising that embedding — 32 numbers — while
the ~20k backbone parameters and the hypernetwork stay frozen.

| | |
| --- | --- |
| Architecture | DeepCGM backbone + Hyper-LoRA cultivar module |
| Embedding dimension | 32 |
| LoRA rank / alpha | 2 / 1.0 |
| Hypernetwork | `Linear(32, 1360)`, no activation |
| Backbone gate matrix | 31 × 649 (five growth gates packed together) |
| Trainable parameters per new cultivar | 32 |
| Sequence length | 365 days from sowing |
| Crop | wheat (winter and spring) |

## Inputs and outputs

**Inputs** per day: `IRRAD` (kJ m⁻² d⁻¹), `TMIN`, `TMAX` (°C), `RAIN` (mm),
`irr` (mm), `fer` (kg N ha⁻¹), `DVS` (–), standardised with the training-set
mean and scale shipped in `input_scaler.json`.

**Outputs** per day: `DVS`, `LAI` (m² m⁻²), `TWLV`, `TWST`, `WSO`, `TAGP`
(kg ha⁻¹).

`DVS` is passed through from the input, not simulated — the model is told the
phenology and predicts growth given it. Treat the DVS column of the output as a
copy of the driver.

## Training data

Four public sources, merged and preprocessed as described in the paper. After
filtering, 606 plot-seasons from 62 cultivars.

| source | contribution | license |
| --- | --- | --- |
| APSIM-Wheat validation data, [APSIMInitiative/ApsimX](https://github.com/APSIMInitiative/ApsimX/tree/master/Tests/Validation/Wheat) | 463 plot-seasons, 39 cultivars | see the ApsimX repository |
| Gaudio et al. (2023), cereal–legume field experiments, [doi:10.5281/zenodo.8081577](https://doi.org/10.5281/zenodo.8081577) | 113 plot-seasons, 15 cultivars | CC BY 4.0 |
| DSSAT-Wheat, [DSSAT/dssat-csm-data](https://github.com/DSSAT/dssat-csm-data) | 20 plot-seasons, 2 cultivars | see the DSSAT repository |
| Cooke et al. (2025), wheat crop water use, [doi:10.7910/DVN/HDKKAL](https://doi.org/10.7910/DVN/HDKKAL) | 10 plot-seasons, 7 cultivars | CC BY 4.0 |

Please credit these sources alongside our paper when you use the released
weights.

The 606 plot-seasons are split by cultivar:

* **Pretraining** — 258 plot-seasons, 45 cultivars. Backbone, hypernetwork and
  embeddings all trainable.
* **Fine-tuning** — 224 plot-seasons, 17 cultivars absent from pretraining.
  Only the embeddings trainable.
* **Testing** — 124 plot-seasons, the same 17 cultivars.

The merged dataset is **not redistributable**, so it is not part of this
repository. Only the weights and one open demo cultivar are released: the 19
Nefer plot-seasons of the Gaudio et al. record, under CC BY 4.0, in
`data/demo_nefer/`.

Both training stages are implemented here — `scripts/pretrain.py` and
`scripts/finetune.py` — but only the second can be run on the data that ship
with the release. Pretraining is reproducible in procedure, not in data:
whoever reassembles the four sources above can rerun it, and
`scripts/pretrain.py --dry-run` reports the architecture and optimiser setup
without any data.

## Training configuration

Adam; 500 epochs for each of pretraining and fine-tuning; full-batch
(batch size 512); learning rate 0.1 for the backbone and 0.01 for the
embeddings and the hypernetwork, decayed by 0.8 every 100 epochs during
pretraining and every 50 during fine-tuning. The loss is a masked, per-variable
weighted MSE — weights `[1, 1, 4, 2, 2, 1]` for
`[DVS, LAI, TWLV, TWST, WSO, TAGP]` — plus the convergence term of
Han et al. (2025).

Neither stage uses an independent validation set for early stopping: the
fine-tuning data are too sparse to split one off, so the epoch with the lowest
*training* loss is kept (the pretraining loss for the backbone, the fine-tuning
loss for the embeddings). See the paper's Section 2.4.

## Model selection

`d_emb ∈ {8, 16, 32} × r ∈ {2, 4, 8}`, ten seeds per cell, selected on the
**fine-tuning** loss so that the test set stayed untouched. The lowest mean
fine-tuning loss was at `d_emb = 32, r = 2`.

The released weights are the single best run within that configuration:

| | |
| --- | --- |
| Seed | 4 |
| Fine-tuning loss | 0.03706 |
| Test loss | 0.07510 |

## Reported performance, and what this checkpoint gives you

The paper's headline numbers come from the **ensemble mean of 10 seeds**: the
ten models' daily predictions are averaged and the metrics computed once on
that average. This checkpoint is one of those ten.

| | mean nRMSE on the test set | cultivars beating WOFOST |
| --- | --- | --- |
| 10-seed ensemble (the paper) | 0.323 | 14 / 17 |
| single seed, on average | 0.350 (sd 0.034; range 0.284–0.400) | ≈ 11.6 / 17 |

So expect this checkpoint to be a little worse than the published figures.
Averaging several fine-tuning runs recovers part of the difference, but the
ensemble in the paper varies the *pretraining* seed as well, which cannot be
reproduced from a single released backbone.

Accuracy is highest for the biomass variables (R² > 0.90 on the fine-tuning
data, and better than WOFOST on the test data) and lowest for LAI and TWLV,
which rise, peak and then senesce, and which carry more measurement
uncertainty.

## Intended use

Simulating wheat growth for cultivars and sites of your own, after fitting a
cultivar embedding on whatever observations you have. It is a research model:
a substitute for calibrating a process-based crop model on sparse data, not a
production forecasting system.

## Limitations

* **Wheat only.** The pretraining data are wheat; nothing about the released
  weights transfers to another crop.
* **Phenology is an input.** Errors in your DVS series propagate directly into
  the biomass trajectories, and the model cannot tell you when a cultivar will
  flower.
* **Water and nitrogen are coarse.** Management enters as cumulative
  irrigation-plus-rain and cumulative nitrogen; there is no soil water balance,
  no soil nitrogen pool and no explicit stress module.
* **No wind or humidity.** Those variables were missing from most of the
  training data and were held at constants (1.5 m s⁻¹, 0.5 kPa) throughout, so
  the model cannot respond to them.
* **Sparse fine-tuning data limit what is constrained.** A cultivar fine-tuned
  without LAI observations will still produce an LAI curve, but its shape is
  inherited from the pretrained cultivars rather than fitted. In the paper,
  cultivars with few fine-tuning observations had markedly less reliable
  per-cultivar accuracy.
* **Geographic coverage** is uneven — mostly Australia, New Zealand and Western
  Europe, with smaller contributions from North America, the Middle East,
  Africa and China.
* **The embedding is not a set of physiological parameters.** Its principal
  components correlate with observed traits (PC1 versus maximum aboveground
  biomass, R² = 0.41), but individual dimensions have no biological meaning.

## Ethical and environmental considerations

The model is trained on published agronomic trials and contains no personal
data. Fine-tuning costs seconds to minutes on a CPU; the cost sits in the
pretraining, which is done once and shipped here.
