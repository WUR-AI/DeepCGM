# Data format

A dataset is a directory holding three CSV files. `data/demo_nefer/` is a
working example — copy its structure.

```
my_data/
├── plots.csv
├── drivers.csv
└── observations.csv
```

---

## `plots.csv` — one row per plot-season

| column | required | description |
| --- | --- | --- |
| `uid` | yes | unique id of the plot-season; joins the other two files |
| `cultivar` | yes | cultivar name; matched against `cultivars.json` in the checkpoint |
| `planting_date` | yes | `YYYY-MM-DD`; day 0 of the 365-day simulation window |
| `split` | no | `finetune` / `test` / anything else; selected with `--split` |
| `latitude`, `longitude` | no | metadata only, not used by the model |
| `harvest_date` | no | metadata only |
| `irrigation_total_mm`, `fertiliser_total_kgN_ha` | no | metadata only |
| `source` | no | provenance |

Any further columns are ignored.

---

## `drivers.csv` — one row per plot-season and day

| column | unit | description |
| --- | --- | --- |
| `uid` | | plot-season id |
| `date` | `YYYY-MM-DD` | calendar day |
| `IRRAD` | kJ m⁻² d⁻¹ | daily global radiation |
| `TMIN` | °C | daily minimum temperature |
| `TMAX` | °C | daily maximum temperature |
| `RAIN` | mm d⁻¹ | precipitation |
| `irr` | mm | irrigation applied on that day (0 on other days) |
| `fer` | kg N ha⁻¹ | nitrogen applied on that day (0 on other days) |
| `DVS` | – | development stage: 0 emergence, 1 anthesis, 2 maturity |

Rows are aligned to a fixed 365-day window starting on `planting_date`. Days
outside the supplied range, or gaps inside it, are handled as follows:

* weather (`IRRAD`, `TMIN`, `TMAX`, `RAIN`) — forward then backward filled;
* management (`irr`, `fer`) — filled with 0;
* `DVS` — forward then backward filled, so a series that stops at maturity is
  simply held at 2.

If a column is still empty after filling, loading fails with an explicit error.

**Pre-planting applications.** The paper aggregates fertiliser and irrigation
applied before sowing onto the planting date, because the model's window starts
there. Do the same.

**About DVS.** DeepCGM is told the phenology and simulates growth given it; it
does not predict development timing itself, and it does not synthesise the DVS
column for you — you must supply it. The paper used DVS from a calibrated WOFOST
run, which is the recommended route if you have
[PCSE](https://github.com/ajwdewit/pcse). The value is 0 at emergence, 1 at
anthesis and 2 at maturity, and must be given for every driver day.

The model freezes growth once DVS reaches 2, so an inaccurate maturity date
directly shifts the end of biomass accumulation.

---

## `observations.csv` — sparse measurements

| column | unit | description |
| --- | --- | --- |
| `uid` | | plot-season id |
| `date` | `YYYY-MM-DD` | measurement day |
| `DVS` | – | observed development stage |
| `LAI` | m² m⁻² | leaf area index |
| `TWLV` | kg ha⁻¹ | total leaf biomass (green + dead) |
| `TWST` | kg ha⁻¹ | total stem biomass |
| `WSO` | kg ha⁻¹ | storage organ (grain) biomass |
| `TAGP` | kg ha⁻¹ | total aboveground biomass |

Leave a cell empty when the variable was not measured; the loss and the metrics
ignore missing values. A plot-season may carry a single measurement, and a
cultivar may lack an entire variable — that is the normal case in field data.

Two conventions from the paper are worth copying:

* organ-specific components are summed into totals before use (e.g. green plus
  dead leaf mass becomes `TWLV`);
* when physiological maturity was not observed, the harvest date or the last
  observation date is recorded as `DVS = 2.0`.

The file may be omitted entirely if you only want to simulate; `predict.py`
then skips the metrics.

---

## Minimal example

`plots.csv`

```csv
uid,cultivar,planting_date,split
field_A_2021,Sculptur,2020-10-22,finetune
field_A_2022,Sculptur,2021-10-19,test
```

`drivers.csv`

```csv
uid,date,IRRAD,TMIN,TMAX,RAIN,irr,fer,DVS
field_A_2021,2020-10-22,4310.0,6.1,14.8,0.0,0.0,40.0,0.0
field_A_2021,2020-10-23,6120.0,4.9,13.2,1.4,0.0,0.0,0.0043
...
```

`observations.csv`

```csv
uid,date,DVS,LAI,TWLV,TWST,WSO,TAGP
field_A_2021,2021-04-14,,3.12,1420,2180,,3600
field_A_2021,2021-06-02,1.0,,,,,
field_A_2021,2021-07-21,2.0,,,,7810,14250
```
