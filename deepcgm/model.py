# -*- coding: utf-8 -*-
"""``DeepCGMGeneric``: the three modules of the paper wired together.

    cultivar id --(embedding table)--> embedding
                --(hypernetwork)-----> LoRA update dW
                --(backbone + dW)----> daily crop state
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .models.DeepCGM_LoRA import DeepCGM
from .models.CultivarEmbedding import CultivarEmbedding
from .models.EmbeddingToLoRA import EmbeddingToLoRA
from .scaling import InputScaler, unscale_outputs

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "deepcgm_generic_e32_r2"


@dataclass
class ModelConfig:
    embedding_dim: int = 32
    hypernetwork_hidden_dim: int = 32
    lora_rank: int = 2
    lora_alpha: float = 1.0
    organ_size: Sequence[int] = (8, 8)
    input_mask: bool = True
    num_cultivar_slots: int = 200
    sequence_length: int = 365
    input_features: Sequence[str] = ("IRRAD", "TMIN", "TMAX", "RAIN", "irr", "fer", "DVS")
    output_features: Sequence[str] = ("DVS", "LAI", "TWLV", "TWST", "WSO", "TAGP")
    output_scale: Sequence[float] = (2.3, 8, 20000, 20000, 20000, 20000)
    loss_weights: Sequence[float] = (1, 1, 4, 2, 2, 1)
    extra: Dict = None

    @classmethod
    def from_json(cls, path) -> "ModelConfig":
        with open(path) as f:
            payload = json.load(f)
        fields = {f for f in cls.__dataclass_fields__ if f != "extra"}
        known = {k: v for k, v in payload.items() if k in fields}
        extra = {k: v for k, v in payload.items() if k not in fields}
        return cls(extra=extra, **known)


class DeepCGMGeneric(nn.Module):
    """The full model. Use :meth:`from_pretrained` to load the released weights."""

    def __init__(self, config: ModelConfig, input_scaler: InputScaler):
        super().__init__()
        self.config = config
        self.input_scaler = input_scaler

        organ_size_C = int(config.organ_size[0])
        # the backbone packs all five gates into one matrix, so the
        # hypernetwork adapts a single (3*organ+7) x (1 + 3*3*organ + (3*organ)^2) layer
        in_features = [organ_size_C * 3 + 7]
        out_features = [1 + organ_size_C * 3 * 3 + (organ_size_C * 3) ** 2]

        self.embeddings = CultivarEmbedding(config.num_cultivar_slots, config.embedding_dim)
        self.hypernetwork = EmbeddingToLoRA(
            embedding_dim=config.embedding_dim,
            in_features_list=in_features,
            out_features_list=out_features,
            r=config.lora_rank,
            alpha=config.lora_alpha,
            hidden_dim=config.hypernetwork_hidden_dim,
        )
        self.backbone = DeepCGM(
            input_scaler=input_scaler,
            organ_size_list=config.organ_size,
            input_mask=config.input_mask,
            r=config.lora_rank,
            alpha=config.lora_alpha,
        )

    # ----------------------------------------------------------------- setup
    @classmethod
    def from_pretrained(cls, directory=DEFAULT_CHECKPOINT, device="cpu",
                        num_cultivar_slots: Optional[int] = None) -> "DeepCGMGeneric":
        directory = Path(directory)
        config = ModelConfig.from_json(directory / "config.json")
        if num_cultivar_slots is not None:
            config.num_cultivar_slots = max(num_cultivar_slots, config.num_cultivar_slots)
        scaler = InputScaler.from_json(directory / "input_scaler.json")

        model = cls(config, scaler)
        model.backbone.load_state_dict(
            torch.load(directory / "backbone.pt", map_location="cpu"), strict=True)
        model.hypernetwork.load_state_dict(
            torch.load(directory / "hypernetwork.pt", map_location="cpu"), strict=True)

        table = torch.load(directory / "cultivar_embeddings.pt", map_location="cpu")["embeddings"]
        with torch.no_grad():
            model.embeddings.embeddings[:table.shape[0]] = table
        return model.to(device)

    def freeze_backbone(self) -> None:
        """Fine-tuning setting of the paper: only the embeddings stay trainable."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.hypernetwork.parameters():
            p.requires_grad_(False)
        self.embeddings.embeddings.requires_grad_(True)

    def reset_embeddings(self, cultivar_ids: Sequence[int]) -> None:
        """Put the given rows back to the zero vector that new cultivars start from."""
        with torch.no_grad():
            self.embeddings.embeddings[list(cultivar_ids)] = 0.0

    # --------------------------------------------------------------- forward
    def forward(self, drivers: torch.Tensor, cultivar_ids: torch.Tensor,
                return_aux: bool = False):
        """``drivers`` is ``[N, 365, 7]`` standardised, ``cultivar_ids`` is ``[N]``.

        Returns the daily state in the model's internal scale.
        """
        embedding = self.embeddings(cultivar_ids)
        lora = self.hypernetwork(embedding)
        return self.backbone(drivers, lora, return_aux=return_aux)

    @torch.no_grad()
    def predict(self, drivers: torch.Tensor, cultivar_ids: torch.Tensor,
                batch_size: int = 64) -> np.ndarray:
        """Simulate seasons and return ``[N, 365, 6]`` in **physical units**."""
        self.eval()
        device = next(self.parameters()).device
        chunks: List[np.ndarray] = []
        for start in range(0, drivers.shape[0], batch_size):
            sl = slice(start, start + batch_size)
            out, _ = self.forward(drivers[sl].to(device), cultivar_ids[sl].to(device))
            chunks.append(out.cpu().numpy())
        return unscale_outputs(np.concatenate(chunks, axis=0), self.config.output_scale)
