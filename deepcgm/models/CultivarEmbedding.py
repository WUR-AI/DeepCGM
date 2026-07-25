# -*- coding: utf-8 -*-
"""Learnable per-cultivar embedding table."""
import torch
from torch import nn


class CultivarEmbedding(nn.Module):
    """A ``[num_cultivars, embedding_dim]`` table of trainable cultivar vectors.

    During fine-tuning this is the *only* module that is optimised: the
    backbone and the hypernetwork stay frozen, so adapting the model to a new
    cultivar costs ``embedding_dim`` trainable parameters (32 in the released
    model).

    New cultivars are initialised to zero, which is how the held-out cultivars
    of the paper entered the fine-tuning stage.
    """

    def __init__(self, num_cultivars: int, embedding_dim: int = 32):
        super().__init__()
        self.num_cultivars = num_cultivars
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Parameter(torch.zeros(num_cultivars, embedding_dim))

    def forward(self, cultivar_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[cultivar_ids]
