# -*- coding: utf-8 -*-
"""Neural modules of DeepCGM-generic."""
from .DeepCGM_LoRA import DeepCGM
from .CultivarEmbedding import CultivarEmbedding
from .EmbeddingToLoRA import EmbeddingToLoRA

__all__ = ["DeepCGM", "CultivarEmbedding", "EmbeddingToLoRA"]
