# Demo dataset — cultivar Nefer

Nineteen plot-seasons of the durum wheat cultivar **Nefer**: 13 for
fine-tuning and 6 held out for testing. This is the dataset behind the
quickstart and behind `docs/figures/fig_nefer_test.png`.

## Source and license

A **modified subset** of:

> Gaudio, Noémie; Mahmoud, Rémi; Bedoussac, Laurent; Justes, Eric;
> Journet, Etienne-Pascal; Naudin, Christophe; Hauggaard-Nielsen, Henrik;
> Jensen, Erik Steen; Pelzer, Elise; Corre-Hellou, Guénaëlle; Kammoun, Bochra;
> Viguier, Loïc; Barillot, Romain; Couëdel, Antoine; Hinsinger, Philippe;
> Casadebaig, Pierre (2023). *A global dataset gathering 37 field experiments
> involving cereal-legume intercrops and their corresponding sole crops.*
> Zenodo. https://doi.org/10.5281/zenodo.8081577

Licensed by its creators under the
**Creative Commons Attribution 4.0 International (CC BY 4.0)** license:
https://creativecommons.org/licenses/by/4.0/

Redistributed here under that license, **not** under this repository's MIT
license. If you use this data, cite the record above and keep the attribution
intact. The original creators do not endorse this repository or its
modifications, and provide the material as-is without warranties.

### Changes made to the original

1. only the 19 plot-seasons of the cultivar Nefer were retained;
2. the tables were reorganised into `plots.csv` / `drivers.csv` /
   `observations.csv` and the columns renamed to this repository's schema
   (see [`docs/data_format.md`](../../docs/data_format.md));
3. **all site information was removed** — coordinates, site names and trial
   identifiers are not part of these files, and the plot-season identifiers
   were replaced by opaque labels `nefer_01` … `nefer_19`;
4. daily weather was aligned and padded to a fixed 365-day window starting at
   the planting date, with gaps filled by forward/backward filling;
5. organ-specific measurements were aggregated into totals (e.g. green plus
   dead leaf mass into `TWLV`);
6. observed phenological stages were converted to continuous DVS values, and
   the harvest or last-observation date was recorded as `DVS = 2.0` where
   physiological maturity was not observed;
7. pre-planting irrigation and fertilisation were aggregated onto the planting
   date;
8. the `DVS` column of `drivers.csv` was **added** — it is not part of the
   original record but comes from the calibrated WOFOST runs of our paper;
9. the `split` column of `plots.csv` was **added**; it reproduces the
   fine-tuning / testing division used in the paper;
10. `wofost_predictions.csv` was **added**; it contains our own WOFOST
    simulations, not original data (see below).

## Contents

| file | rows | what |
| --- | --- | --- |
| `plots.csv` | 19 | plot-season id, cultivar, planting date, split |
| `drivers.csv` | 6 935 | daily weather, management and DVS |
| `observations.csv` | 139 | sparse field measurements |
| `wofost_predictions.csv` | 6 935 | daily WOFOST simulation, for comparison |

`wofost_predictions.csv` holds the output of the calibrated WOFOST 8.1 runs
reported in the paper (Case 1), in physical units, indexed by `uid` and days
after planting. It is provided so that the demo figure can show the
process-based baseline; the WOFOST model itself is not part of this
repository and is available through [PCSE](https://github.com/ajwdewit/pcse).

## Why Nefer

It makes an honest demonstration of adapting to a genuinely new cultivar:

* the backbone and hypernetwork **never saw Nefer** during pretraining, so
  fitting its embedding is the same problem you face with your own data;
* the fine-tuning seasons were sown in **2005–2006** and the test seasons in
  **2011–2012**, five or more growing seasons apart, so the test weather is
  independent rather than a rerun of what was fitted.

One caveat: all Nefer plot-seasons come from the same experimental station, in
fields a few hundred metres apart. The separation is **temporal, not spatial**,
and the plots share a climate.

## What the test split can and cannot tell you

The six test plot-seasons carry 78 measurements, but they are unevenly spread:

| variable | observations in the test split |
| --- | --- |
| LAI | 8 |
| TWLV | **0** |
| TWST | 8 |
| WSO | 6 |
| TAGP | 22 |

Leaf biomass is never measured in the test seasons. The model still simulates
it — the curve in the figure comes from what the pretrained backbone learnt on
other cultivars plus the fine-tuned embedding — but nothing here validates it.
That is the normal state of field data, and the reason to read the columns with
many observations more confidently than the ones resting on a handful of
points. Your own data will very likely have gaps like this too.
