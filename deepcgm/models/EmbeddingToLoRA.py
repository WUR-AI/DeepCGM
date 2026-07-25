# -*- coding: utf-8 -*-
"""Hypernetwork that turns a cultivar embedding into LoRA weight updates."""
from typing import List, Sequence

import torch
from torch import nn


class EmbeddingToLoRA(nn.Module):
    """Generate the LoRA matrices ``A`` and ``B`` for each adapted layer.

    For a target layer with ``in_features`` x ``out_features`` weights and rank
    ``r``, a single linear map produces ``in_features * r + r * out_features``
    numbers, which are reshaped into ``A`` and ``B`` and multiplied to give the
    low-rank update ``dW = A @ B``.

    The released model adapts one layer (the packed growth gates,
    31 x 649) at rank 2, so the hypernetwork is a single
    ``Linear(embedding_dim, 1360)``.
    """

    def __init__(self, embedding_dim: int, in_features_list: Sequence[int],
                 out_features_list: Sequence[int], r: int, alpha: float = 1.0,
                 hidden_dim: int = 32):
        super().__init__()
        assert len(in_features_list) == len(out_features_list), \
            "in_features_list and out_features_list must have the same length."

        self.embedding_dim = embedding_dim
        self.in_features_list = list(in_features_list)
        self.out_features_list = list(out_features_list)
        self.r = r
        self.alpha = alpha
        # kept for checkpoint compatibility; the released hypernetwork is a
        # single linear layer without activations (see the paper, Section 2.4)
        self.hidden_dim = hidden_dim
        self.scaling = self.alpha / self.r if self.r > 0 else 1.0

        self.hypernetworks = nn.ModuleList(
            nn.Sequential(nn.Linear(embedding_dim, in_feat * r + r * out_feat))
            for in_feat, out_feat in zip(self.in_features_list, self.out_features_list)
        )

    def forward(self, embeddings: torch.Tensor) -> List[torch.Tensor]:
        """``[batch, embedding_dim]`` -> list of ``[batch, in_feat, out_feat]``."""
        batch_size = embeddings.shape[0]
        updates = []
        for hypernet, in_feat, out_feat in zip(self.hypernetworks,
                                               self.in_features_list,
                                               self.out_features_list):
            params = hypernet(embeddings)
            a_size = in_feat * self.r
            b_size = self.r * out_feat
            A = params[:, :a_size].view(batch_size, in_feat, self.r)
            B = params[:, a_size:a_size + b_size].view(batch_size, self.r, out_feat)
            updates.append(torch.bmm(A, B))
        return updates
