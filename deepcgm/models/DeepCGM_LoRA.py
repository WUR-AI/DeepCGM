# -*- coding: utf-8 -*-
"""DeepCGM backbone with LoRA-adapted growth gates.

This is the differentiable crop growth model of Han et al. (2025), extended so
that the weight matrix of each growth gate can receive a cultivar-specific
low-rank update produced by the hypernetwork (see ``EmbeddingToLoRA.py``).

The module structure and parameter names are kept identical to the code used
for the paper so that the released checkpoints load with ``strict=True``.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Converter(nn.Module):
    """Element-wise learnable scaling with a non-negative output."""

    def __init__(self, var_num: int, value: float):
        super().__init__()
        self.var_num = var_num
        self.par = nn.Parameter(torch.ones(self.var_num) * value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x * self.par)


class Normalize(nn.Module):
    """L1 normalisation over the last dimension."""

    def __init__(self, p: float = 1, dim: int = -1, eps: float = 1e-12):
        super().__init__()
        self.p, self.dim, self.eps = p, dim, eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=self.p, dim=self.dim, eps=self.eps)


class LoRALinear(nn.Module):
    """Linear layer whose weight can be shifted by a per-sample LoRA update.

    ``lora_AB`` has shape ``[batch, in_features, out_features]`` and is produced
    by the hypernetwork from the cultivar embedding. When it is ``None`` the
    layer behaves as a plain shared linear layer.
    """

    def __init__(self, in_features: int, out_features: int, r: int = 6, alpha: float = 1.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.bias = nn.Parameter(torch.randn(out_features))
        self.scaling = self.alpha / self.r if self.r > 0 else 1.0

    def forward(self, x: torch.Tensor, lora_AB: Optional[torch.Tensor],
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lora_AB is not None:
            weight = self.weight.unsqueeze(0) + lora_AB * self.scaling
        else:
            weight = self.weight.unsqueeze(0).expand(x.shape[0], -1, -1)

        if mask is not None:
            weight = weight * mask.unsqueeze(0)

        if x.dim() == 3:  # [batch, time, in_features]
            return torch.bmm(x, weight) + self.bias
        return torch.bmm(x.unsqueeze(1), weight).squeeze(1) + self.bias


class CombinedGate(nn.Module):
    """The five DeepCGM growth gates packed into a single LoRA linear layer."""

    def __init__(self, input_num: int, gate_shapes: Sequence[Tuple[int, int]],
                 activations, normalisers, i_prior=None, r: int = 6, alpha: float = 1.0):
        super().__init__()
        self.input_num = input_num
        self.gate_shapes = gate_shapes
        self.activations = activations
        self.normalisers = normalisers
        self.gate_nums = [int(np.prod(shape)) for shape in gate_shapes]

        self.lora_linear = LoRALinear(input_num, sum(self.gate_nums), r=r, alpha=alpha)

        if i_prior is not None:
            # not persistent: the mask is derived from the config, not learned,
            # so it must not appear in the released state_dict
            self.register_buffer("i_combine_prior", torch.cat(i_prior, dim=1),
                                 persistent=False)
        else:
            self.i_combine_prior = None

    def forward(self, x: torch.Tensor, lora_AB: Optional[torch.Tensor]) -> List[torch.Tensor]:
        gates = self.lora_linear(x, lora_AB, mask=self.i_combine_prior)
        chunks = torch.split(gates, self.gate_nums, dim=-1)
        chunks = [c.view((-1, c.shape[1], *shape)) for c, shape in zip(chunks, self.gate_shapes)]
        return [norm(act(c)) for c, act, norm in zip(chunks, self.activations, self.normalisers)]


class DeepCGM(nn.Module):
    """Differentiable crop growth model.

    Parameters
    ----------
    input_scaler:
        A :class:`deepcgm.scaling.InputScaler`. The model receives standardised
        drivers but needs a few of them in physical units (radiation, rain,
        irrigation, fertiliser and DVS), so it un-scales those internally.
    organ_size_list:
        ``[organ_size_C, organ_size_D]``; only the first entry is used by this
        carbon-only version. Each organ (leaf / stem / storage organ) is
        represented by ``organ_size_C`` carbon pools.
    input_mask:
        Apply the structural prior that blocks weather inputs from the
        redistribution gate.
    r, alpha:
        LoRA rank and scaling factor; must match the hypernetwork.
    """

    OUTPUTS = ("DVS", "LAI", "TWLV", "TWST", "WSO", "TAGP")

    def __init__(self, input_scaler, organ_size_list: Sequence[int] = (8, 8),
                 input_mask: bool = True, r: int = 2, alpha: float = 1.0):
        super().__init__()
        self.name = "DeepCGM_LoRA"

        organ_size_C = int(organ_size_list[0])
        self.dim_segment_C = torch.cumsum(
            torch.tensor([0, organ_size_C, organ_size_C, organ_size_C]), 0)
        self.C_cell_num = organ_size_C * 3

        # DVS, IRRAD, TMAX, TMIN, fer, irr, time step
        self.aux_num = 7
        self.input_num_C = self.C_cell_num + self.aux_num

        self.register_buffer("C_init", torch.zeros(self.C_cell_num), persistent=False)

        self.C_assimilate_shape = (1, 1)
        self.C_partitation_shape = (1, self.C_cell_num)
        self.C_consuming_shape = (1, self.C_cell_num)
        self.C_redistribution_shape = (self.C_cell_num, self.C_cell_num)

        gate_shapes_C = [
            self.C_assimilate_shape,       # interception / assimilation
            self.C_partitation_shape,      # partitioning
            self.C_consuming_shape,        # growth respiration
            self.C_consuming_shape,        # maintenance respiration
            self.C_redistribution_shape,   # redistribution
        ]
        normalisers = [nn.Sigmoid(), Normalize(), nn.Sigmoid(), nn.Sigmoid(), Normalize()]
        activations = [nn.Identity(), nn.ReLU(), nn.Identity(), nn.Identity(), nn.ReLU()]

        self.C2A = Converter(self.C_cell_num, 0.001)   # carbon -> leaf area
        self.G2Y = Converter(organ_size_C, 1)          # storage organ -> yield
        self.C2B = nn.Linear(1, 1)

        i_prior_C = None
        if input_mask:
            priors = [torch.ones(self.input_num_C, int(np.prod(shape)))
                      for shape in gate_shapes_C]
            # weather and cumulative management may not drive redistribution
            for offset in range(1, 6):
                priors[-1][self.C_cell_num + offset, :] = 0
            i_prior_C = priors

        self.combined_gate_C = CombinedGate(
            self.input_num_C, gate_shapes_C, activations, normalisers,
            i_prior_C, r=r, alpha=alpha)

        self.input_scaler = input_scaler
        self.register_buffer("_inp_mean", torch.as_tensor(input_scaler.mean, dtype=torch.float32),
                             persistent=False)
        self.register_buffer("_inp_scale", torch.as_tensor(input_scaler.scale, dtype=torch.float32),
                             persistent=False)

    # ------------------------------------------------------------------ core
    def _unscale(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        return x * self._inp_scale[idx] + self._inp_mean[idx]

    def rate(self, C_cell, x, C_potential, timestep, lora_AB, gate):
        (C_assimilate_ratio, C_partitation_mat, C_growResp_mat,
         C_mainResp_mat, C_redistribution_mat) = gate(x, lora_AB)

        C_in = C_assimilate_ratio.squeeze(-2) * C_potential * timestep
        C_mainResp = torch.mul(C_cell.unsqueeze(-2), C_mainResp_mat).squeeze(-2) * timestep
        C_net = F.relu(C_in - C_mainResp.sum(-1, keepdim=True))
        C_grow = torch.matmul(C_net.unsqueeze(-2), C_partitation_mat).squeeze(-2)
        C_growResp = C_grow.unsqueeze(-2) * C_growResp_mat
        C_grow_net = C_grow.unsqueeze(-2) - C_growResp
        C_cell = C_cell + C_grow_net.squeeze(-2)
        C_cell = torch.matmul(C_cell.unsqueeze(-2), C_redistribution_mat).squeeze(-2)
        return C_cell

    def preprocessing(self, X: torch.Tensor):
        """Split the standardised driver tensor into the pieces the gates need.

        ``X`` is ``[batch, time, 7]`` ordered as
        ``[IRRAD, TMIN, TMAX, RAIN, irr, fer, DVS]``.
        """
        batch_size = X.shape[0]
        C_cell = self.C_init.repeat(batch_size, 1).unsqueeze(-2)

        rad, tmin, tmax = X[:, :, [0]], X[:, :, [1]], X[:, :, [2]]
        dvs = X[:, :, [6]]

        rai_raw = self._unscale(X[:, :, 3], 3)
        irr_raw = self._unscale(X[:, :, 4], 4)
        fer_raw = self._unscale(X[:, :, 5], 5)

        N_cum = torch.cumsum(fer_raw, -1).unsqueeze(-1) / 200
        W_cum = (torch.cumsum(rai_raw, -1) + torch.cumsum(irr_raw, -1)).unsqueeze(-1) / 500

        # radiation -> potential carbon assimilation (kg/ha, scaled by 1/20000)
        rad_raw = self._unscale(rad, 0)
        FRPAR, eff, scale_par, CO2_2_C = 0.5, 0.45, 1 / 20000, 12 / 44
        C_potential = rad_raw * (scale_par * FRPAR * eff / 3.6 * CO2_2_C)

        AUX = torch.cat([rad, tmax, tmin, N_cum, W_cum], -1)
        return C_cell, AUX, C_potential, dvs

    def forward(self, X: torch.Tensor, lora_AB_list: Optional[List[torch.Tensor]] = None,
                return_aux: bool = False):
        """Simulate a season.

        Returns ``[batch, time, 6]`` in **scaled** units, i.e. the physical
        values divided by ``config['output_scale']``.
        """
        C_cell, AUX, C_potential, dvs = self.preprocessing(X)
        lora_AB_C = lora_AB_list[0] if lora_AB_list is not None else None

        C_cell_all = torch.empty(X.shape[0], X.shape[1], self.C_cell_num,
                                 device=X.device, dtype=X.dtype)
        C_cell_ini = C_cell

        dvs_phys_all = self._unscale(dvs, 6)
        for t in range(X.shape[1]):
            # growth is frozen once the crop has reached maturity (DVS >= 2)
            alive = (dvs_phys_all[:, [t], :] < 1.99).squeeze(-1)
            if alive.any():
                x = torch.cat([C_cell, dvs[:, [t], :], AUX[:, [t], :],
                               torch.ones_like(C_cell[:, :, [0]])], -1)
                C_cell_new = self.rate(C_cell, x, C_potential[:, [t], :], 1,
                                       lora_AB_C, self.combined_gate_C)
                C_cell = torch.where(alive.unsqueeze(-1), C_cell_new, C_cell)
            C_cell_all[:, t, :] = C_cell.squeeze(-2)

        # convergence branch: re-run each day in sub-steps from the previous state
        C_conv = torch.cat([C_cell_ini, C_cell_all[:, :-1]], dim=1)
        sub_step = 2
        step = 1 / sub_step
        for _ in range(sub_step):
            X_conv = torch.cat([C_conv, dvs, AUX, torch.full_like(C_conv[:, :, [0]], step)], -1)
            C_conv = self.rate(C_conv, X_conv, C_potential, step, lora_AB_C, self.combined_gate_C)

        seg = self.dim_segment_C
        dvs_out = dvs_phys_all / 2.3
        pai = torch.sum(self.C2A(C_cell_all)[:, :, :seg[3]], dim=2, keepdim=True)
        lea = torch.sum(C_cell_all[:, :, seg[0]:seg[1]], dim=2, keepdim=True) / 0.45
        ste = torch.sum(C_cell_all[:, :, seg[1]:seg[2]], dim=2, keepdim=True) / 0.45
        gra = torch.sum(C_cell_all[:, :, seg[2]:seg[3]], dim=2, keepdim=True) / 0.45
        agb = lea + ste + gra

        out = torch.cat([dvs_out, pai, lea, ste, gra, agb], 2)
        if return_aux:
            return out, (C_cell_all, C_conv)
        return out, None
