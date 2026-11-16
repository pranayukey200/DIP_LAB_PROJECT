"""
TransformTrust: Interpretability and Reliability for Vision Transformers
========================================================================

A toolkit for analyzing attention consistency, perturbation robustness,
and trust scores across Vision Transformer representations.
"""

from .model import (
    FALLBACK_MODEL,
    GRID_SIZE,
    IMAGE_SIZE,
    NUM_PATCHES,
    NUM_TOKENS,
    PATCH_SIZE,
    PRIMARY_MODEL,
    get_device,
    get_label_name,
    get_model_name,
    load_model,
    predict,
    predict_batch,
)

__all__ = [
    "PRIMARY_MODEL",
    "FALLBACK_MODEL",
    "IMAGE_SIZE",
    "PATCH_SIZE",
    "GRID_SIZE",
    "NUM_PATCHES",
    "NUM_TOKENS",
    "get_device",
    "get_model_name",
    "get_label_name",
    "load_model",
    "predict",
    "predict_batch",
]
