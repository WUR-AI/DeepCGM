# -*- coding: utf-8 -*-
"""DeepCGM-generic: a pretrained, cultivar-adaptable deep learning crop model.

Typical use::

    from deepcgm import DeepCGMGeneric, CultivarIndex, load_dataset

    model = DeepCGMGeneric.from_pretrained()
    index = CultivarIndex.from_checkpoint("checkpoints/deepcgm_generic_e32_r2/cultivars.json")
    batch = load_dataset("data/demo_nefer", model.input_scaler,
                         model.config.output_scale, index)
    prediction = model.predict(batch.drivers, batch.cultivar_ids)
"""
from .data import CultivarIndex, SeasonBatch, load_dataset
from .ensemble import Ensemble
from .model import DeepCGMGeneric, ModelConfig
from .scaling import InputScaler, scale_outputs, unscale_outputs

__all__ = [
    "CultivarIndex",
    "Ensemble",
    "DeepCGMGeneric",
    "InputScaler",
    "ModelConfig",
    "SeasonBatch",
    "load_dataset",
    "scale_outputs",
    "unscale_outputs",
]

__version__ = "1.0.0"
