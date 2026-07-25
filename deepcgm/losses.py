# -*- coding: utf-8 -*-
"""Objective used for pretraining and fine-tuning.

``Loss_total = Loss_fitting + Loss_cg``

``Loss_fitting`` is a masked, per-variable weighted MSE, so plot-seasons that
lack a variable simply do not contribute to it. ``Loss_cg`` is the convergence
loss of Han et al. (2025): it penalises the difference between the carbon state
after one daily step and after two half steps, which pushes the model towards a
stable carbon balance on days without ground truth.
"""
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

#: weights for [DVS, LAI, TWLV, TWST, WSO, TAGP]
DEFAULT_LOSS_WEIGHTS = (1, 1, 4, 2, 2, 1)
CONVERGENCE_SCALE = 100000.0


def fitting_loss(prediction: torch.Tensor, target: torch.Tensor,
                 weights: Sequence[float] = DEFAULT_LOSS_WEIGHTS,
                 denominators: Optional[Sequence[float]] = None) -> torch.Tensor:
    """Weighted MSE over observed points only.

    ``target`` may contain NaN; those entries are ignored.

    Each variable's squared error is divided by the number of observations of
    that variable **in the batch**. Pass ``denominators`` to divide by a fixed
    count instead — see :func:`reference_denominators`.
    """
    w = torch.as_tensor(weights, device=prediction.device, dtype=prediction.dtype)
    mask = ~torch.isnan(target)
    safe_target = torch.nan_to_num(target, nan=0.0)

    squared = F.mse_loss(prediction, safe_target, reduction="none")
    squared = torch.where(mask, squared, torch.zeros_like(squared))
    if denominators is None:
        counts = mask.sum(dim=(0, 1)).clamp(min=1).to(prediction.dtype)
    else:
        counts = torch.as_tensor(denominators, device=prediction.device,
                                 dtype=prediction.dtype).clamp(min=1)
    per_feature = squared.sum(dim=(0, 1)) / counts
    return (per_feature * w).sum()


def convergence_loss(target: torch.Tensor, carbon_state: torch.Tensor,
                     carbon_state_substeps: torch.Tensor,
                     scale: float = CONVERGENCE_SCALE,
                     denominator: Optional[float] = None) -> torch.Tensor:
    """Discrepancy between the one-step and the multi-sub-step carbon state.

    It is evaluated on the days that carry an observation (the first output
    column, DVS, is used as the "this day was measured" indicator). The average
    runs over observed days **times carbon pools**, so a fixed ``denominator``
    must include that factor.
    """
    mask = ~torch.isnan(target[:, :, 0])              # [batch, time]
    if not mask.any():
        return torch.zeros((), device=carbon_state.device, dtype=carbon_state.dtype)
    mask = mask.unsqueeze(-1).expand_as(carbon_state)
    squared = F.mse_loss(carbon_state_substeps, carbon_state, reduction="none")
    selected = squared.masked_select(mask)
    if denominator is None:
        return selected.mean() * scale
    return selected.sum() / max(float(denominator), 1.0) * scale


def total_loss(prediction, target, carbon_state, carbon_state_substeps,
               weights: Sequence[float] = DEFAULT_LOSS_WEIGHTS,
               use_convergence: bool = True,
               denominators: Optional[Mapping[str, object]] = None):
    """Returns ``(total, fitting, convergence)``.

    ``denominators`` is the optional mapping returned by
    :func:`reference_denominators`.
    """
    fit_counts = denominators.get("fitting") if denominators else None
    cg_count = denominators.get("convergence") if denominators else None
    fit = fitting_loss(prediction, target, weights, denominators=fit_counts)
    cg = (convergence_loss(target, carbon_state, carbon_state_substeps,
                           denominator=cg_count)
          if use_convergence else torch.zeros_like(fit))
    return fit + cg, fit, cg


def reference_denominators(config) -> Optional[Dict[str, object]]:
    """The loss denominators of the batch a checkpoint was fine-tuned on.

    Fine-tuning normalises each variable by the number of observations **in the
    batch**, so running one cultivar on its own changes those denominators — and
    it changes them by a different factor per variable, which tilts the gradient
    towards the variables that cultivar happens to have fewest of. Adam cannot
    undo that, because it is a change of direction and not of scale.

    Cultivar embeddings are independent rows of one table, so no other cultivar
    contributes gradient to the row being fitted. Reusing the original
    denominators therefore makes a single-cultivar run follow the same
    trajectory as the full multi-cultivar fine-tune that produced the released
    weights. Returns ``None`` when the checkpoint does not record them.
    """
    counts = (config.extra or {}).get("reference_loss_counts")
    if not counts:
        return None
    order = list(config.output_features)
    fitting = counts.get("fitting", {})
    return {
        "fitting": [float(fitting[name]) for name in order],
        "convergence": float(counts["convergence"]),
    }
