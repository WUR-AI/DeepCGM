# -*- coding: utf-8 -*-
"""End-to-end checks that the released checkpoint loads and behaves sensibly."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deepcgm import CultivarIndex, DeepCGMGeneric, Ensemble, load_dataset  # noqa: E402
from deepcgm.losses import fitting_loss  # noqa: E402

CHECKPOINT = ROOT / "checkpoints" / "deepcgm_generic_e32_r2"
DEMO = ROOT / "data" / "demo_nefer"


@pytest.fixture(scope="module")
def model():
    return DeepCGMGeneric.from_pretrained(CHECKPOINT)


@pytest.fixture(scope="module")
def batch(model):
    index = CultivarIndex.from_checkpoint(CHECKPOINT / "cultivars.json")
    return load_dataset(DEMO, model.input_scaler, model.config.output_scale, index)


def test_checkpoint_loads_strictly(model):
    assert model.config.embedding_dim == 32
    assert model.config.lora_rank == 2
    assert model.embeddings.embeddings.shape == (200, 32)


def test_dataset_shapes(batch):
    assert len(batch) == 19
    assert batch.drivers.shape == (19, 365, 7)
    assert batch.targets.shape == (19, 365, 6)
    assert not torch.isnan(batch.drivers).any()


def test_predictions_are_physically_plausible(model, batch):
    prediction = model.predict(batch.drivers, batch.cultivar_ids)
    assert prediction.shape == (19, 365, 6)
    assert np.isfinite(prediction).all()

    lai, tagp = prediction[:, :, 1], prediction[:, :, 5]
    assert (lai >= 0).all() and lai.max() < 15
    assert (tagp >= 0).all() and 2000 < tagp[:, -1].min() and tagp[:, -1].max() < 40000
    # aboveground biomass is the sum of the organs
    organs = prediction[:, :, 2] + prediction[:, :, 3] + prediction[:, :, 4]
    assert np.allclose(organs, tagp, rtol=1e-4)
    # growth stops once the crop reaches maturity
    mature = prediction[:, :, 0] >= 1.99
    for i in range(prediction.shape[0]):
        if mature[i].any():
            frozen = tagp[i][mature[i]]
            assert np.allclose(frozen, frozen[0], rtol=1e-5)


def test_embedding_changes_the_simulation(model, batch):
    """The same weather under two cultivars must not give the same crop."""
    fitted = batch.cultivar_ids[:1]
    other = torch.tensor([int(fitted) + 1])
    baseline = model.predict(batch.drivers[:1], fitted)
    alternative = model.predict(batch.drivers[:1], other)
    assert not np.allclose(baseline, alternative)


def test_ensemble_mean_sits_between_its_members(batch):
    ensemble = Ensemble.from_directory(ROOT / "checkpoints" / "ensemble_e32_r2")
    assert len(ensemble) == 10
    members = ensemble.predict(batch.drivers[:2], batch.cultivar_ids[:2], reduce=None)
    mean = ensemble.predict(batch.drivers[:2], batch.cultivar_ids[:2])
    assert members.shape == (10, 2, 365, 6)
    assert np.allclose(members.mean(axis=0), mean)
    tagp = members[..., 5]
    assert (tagp.min(axis=0) <= mean[..., 5] + 1e-6).all()
    assert (mean[..., 5] <= tagp.max(axis=0) + 1e-6).all()
    # the members really are different models
    assert tagp.std(axis=0).max() > 1.0


def test_fitting_loss_ignores_missing_observations():
    prediction = torch.zeros(2, 5, 6)
    target = torch.full((2, 5, 6), float("nan"))
    target[0, 0, 5] = 1.0
    loss = fitting_loss(prediction, target)
    assert torch.isfinite(loss) and float(loss) == pytest.approx(1.0)


def test_reference_denominators_are_recorded_and_used(model):
    """The counts that let a single-cultivar run reproduce a released embedding."""
    from deepcgm.losses import fitting_loss, reference_denominators

    denominators = reference_denominators(model.config)
    assert denominators is not None
    # the Finetuning column of Table 3 in the paper
    assert denominators["fitting"] == [1202.0, 223.0, 219.0, 246.0, 294.0, 562.0]
    # the convergence loss averages over observed days x carbon pools
    assert denominators["convergence"] == 1202 * 24

    prediction = torch.zeros(2, 5, 6)
    target = torch.full((2, 5, 6), float("nan"))
    target[0, 0, 5] = 1.0
    own_counts = fitting_loss(prediction, target)
    pinned = fitting_loss(prediction, target, denominators=denominators["fitting"])
    # one TAGP observation: 1/1 with the batch's own count, 1/562 with the pinned one
    assert float(own_counts) == pytest.approx(1.0)
    assert float(pinned) == pytest.approx(1.0 / 562)
