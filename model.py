"""
TransformTrust: Model Loading and Inference Module
==================================================

Handles loading, caching, device management, and inference for Vision Transformers
(ViT and DeiT). Automatically configures attention extraction with eager implementation
and provides single-image and batched predictions with multi-layer attention maps.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

logger = logging.getLogger(__name__)

# Model identifiers
PRIMARY_MODEL = "google/vit-base-patch16-224"       # Preferred for GPU environments
FALLBACK_MODEL = "facebook/deit-tiny-patch16-224"   # Lightweight fallback for CPU

# Vision Transformer Patch Specifications (16x16 patches on 224x224 images)
IMAGE_SIZE = 224
PATCH_SIZE = 16
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE   # 14 x 14
NUM_PATCHES = GRID_SIZE * GRID_SIZE    # 196 spatial patches
NUM_TOKENS = NUM_PATCHES + 1           # 1 CLS token + 196 patch tokens = 197 tokens


def get_device() -> torch.device:
    """Auto-detect and return the active device (CUDA if available, else CPU).

    Returns:
        torch.device: CUDA device if a GPU is available, otherwise CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_model_name(device: Optional[Union[str, torch.device]] = None) -> str:
    """Determine the model identifier based on device capability.

    Selects google/vit-base-patch16-224 when running on CUDA (GPU),
    and falls back to facebook/deit-tiny-patch16-224 when running on CPU.

    Args:
        device: Optional target device. If None, auto-detected via get_device().

    Returns:
        str: Hugging Face model repository ID.
    """
    if device is None:
        target_device = get_device()
    else:
        target_device = torch.device(device)

    if target_device.type == "cuda":
        return PRIMARY_MODEL
    return FALLBACK_MODEL


def get_label_name(config: Any, class_id: int) -> str:
    """Safely retrieve human-readable class name from model configuration.

    Handles ImageNet 1000-class label mappings with int or string keys.

    Args:
        config: Model configuration object (PretrainedConfig).
        class_id: Numerical class index.

    Returns:
        str: Human-readable class name, or 'Class <id>' if not found.
    """
    id2label = getattr(config, "id2label", None)
    if not id2label or not isinstance(id2label, dict):
        return f"Class {class_id}"

    if class_id in id2label:
        return str(id2label[class_id])

    str_id = str(class_id)
    if str_id in id2label:
        return str(id2label[str_id])

    return f"Class {class_id}"


def _load_model_uncached(model_name: str, device_str: str) -> Tuple[Any, Any]:
    """Load model and processor directly from Hugging Face Hub or local cache.

    Attempts to load with attn_implementation='eager' for robust self-attention
    extraction, falling back gracefully if not supported by the environment.

    Args:
        model_name: Model identifier or local checkpoint path.
        device_str: Device string ('cuda' or 'cpu').

    Returns:
        tuple: (model in eval mode on target device, processor)
    """
    device = torch.device(device_str)

    # 1. Load Image Processor
    logger.info("Loading image processor for %s...", model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)

    # 2. Load Model with Attention Extraction Capability
    logger.info("Loading model %s onto %s...", model_name, device_str)
    try:
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            output_attentions=True,
            attn_implementation="eager",
        )
    except Exception as exc:
        logger.warning(
            "Loading %s with attn_implementation='eager' raised: %s. "
            "Falling back to default implementation with output_attentions=True.",
            model_name,
            exc,
        )
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            output_attentions=True,
        )

    # Ensure output_attentions flag is explicitly enabled
    model.config.output_attentions = True

    # Place model on target device and set to evaluation mode
    model.to(device)
    model.eval()

    logger.info("Successfully loaded and initialized %s on %s.", model_name, device_str)
    return model, processor


# Streamlit caching integration with graceful fallback to functools.lru_cache
try:
    import streamlit as st

    @st.cache_resource(show_spinner="Loading Vision Transformer model...")
    def _cached_load(model_name: str, device_str: str) -> Tuple[Any, Any]:
        return _load_model_uncached(model_name, device_str)

except Exception:
    @functools.lru_cache(maxsize=4)
    def _cached_load(model_name: str, device_str: str) -> Tuple[Any, Any]:
        return _load_model_uncached(model_name, device_str)


def load_model(
    model_name: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[Any, Any]:
    """Load and cache the Vision Transformer model and image processor.

    Utilizes Streamlit's cache_resource (or LRU cache) to avoid redundant loading.

    Args:
        model_name: Optional model identifier. If None, auto-selected based on device.
        device: Optional target device. If None, auto-detected via get_device().

    Returns:
        tuple: (model, processor) ready for inference.
    """
    if device is None:
        target_device = get_device()
    else:
        target_device = torch.device(device)

    if model_name is None:
        model_name = get_model_name(target_device)

    device_str = str(target_device)

    try:
        return _cached_load(model_name, device_str)
    except Exception as exc:
        logger.warning(
            "Cached loader encountered an error (%s). Falling back to direct load.",
            exc,
        )
        return _load_model_uncached(model_name, device_str)


def predict_batch(
    images: List[Union[Image.Image, np.ndarray, Any]],
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> List[Dict[str, Any]]:
    """Perform batched inference on a list of images.

    Extracts predictions, confidence scores, top-5 classes, raw logits,
    softmax probability distributions, and multi-layer attention weight tensors.

    Args:
        images: List of PIL Images or numpy ndarrays.
        model: Optional pre-loaded model. If None, automatically loaded.
        processor: Optional pre-loaded processor. If None, automatically loaded.
        device: Optional torch device. If None, detected from model or get_device().

    Returns:
        list of dicts, each containing:
            - 'predicted_class' (str): Top-1 class name from ImageNet.
            - 'class_id' (int): Top-1 class index (0-999).
            - 'confidence' (float): Top-1 softmax probability.
            - 'top5' (list[tuple[str, float]]): Top 5 classes and their probabilities.
            - 'logits' (torch.Tensor): 1D tensor of raw logits on CPU.
            - 'probabilities' (torch.Tensor): 1D tensor of softmax probabilities on CPU.
            - 'attentions' (list[torch.Tensor]): Attention weight tensors per layer on CPU,
              each with shape [1, num_heads, 197, 197].
    """
    if not images:
        return []

    # Determine execution device
    if device is None:
        if model is not None and hasattr(model, "device"):
            target_device = model.device
        else:
            target_device = get_device()
    else:
        target_device = torch.device(device)

    # Ensure model and processor are initialized
    if model is None or processor is None:
        loaded_model, loaded_processor = load_model(device=target_device)
        model = model or loaded_model
        processor = processor or loaded_processor

    # Guarantee eval mode and placement
    model.eval()
    model.to(target_device)

    # Convert inputs to PIL RGB Images
    pil_images: List[Image.Image] = []
    for img in images:
        if isinstance(img, Image.Image):
            pil_images.append(img.convert("RGB"))
        elif isinstance(img, np.ndarray):
            pil_images.append(Image.fromarray(img).convert("RGB"))
        else:
            pil_images.append(Image.fromarray(np.array(img)).convert("RGB"))

    # Preprocess images through the transformer processor
    inputs = processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    # Run inference with gradients disabled
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    logits_batch = outputs.logits  # Shape: [batch_size, num_classes]
    probs_batch = F.softmax(logits_batch, dim=-1)  # Shape: [batch_size, num_classes]
    raw_attentions = outputs.attentions  # Tuple of num_layers tensors: [batch_size, heads, tokens, tokens]

    batch_size = len(pil_images)
    results: List[Dict[str, Any]] = []

    for i in range(batch_size):
        logits_i = logits_batch[i].detach().cpu()
        probs_i = probs_batch[i].detach().cpu()

        # Compute top 5 candidates
        k = min(5, probs_i.shape[-1])
        top_probs, top_indices = torch.topk(probs_i, k=k)

        top5: List[Tuple[str, float]] = [
            (get_label_name(model.config, int(idx.item())), float(prob.item()))
            for idx, prob in zip(top_indices, top_probs)
        ]

        top1_idx = int(top_indices[0].item())
        top1_prob = float(top_probs[0].item())
        top1_label = get_label_name(model.config, top1_idx)

        # Slice attention tensors for item i, keeping [1, num_heads, seq_len, seq_len]
        if raw_attentions is not None:
            attentions_i: List[torch.Tensor] = [
                layer_att[i : i + 1].detach().cpu() for layer_att in raw_attentions
            ]
        else:
            attentions_i = []

        results.append(
            {
                "predicted_class": top1_label,
                "class_id": top1_idx,
                "confidence": top1_prob,
                "top5": top5,
                "logits": logits_i,
                "probabilities": probs_i,
                "attentions": attentions_i,
            }
        )

    return results


def predict(
    image: Union[Image.Image, np.ndarray, Any],
    model: Optional[Any] = None,
    processor: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Run inference on a single image and extract prediction details and attention maps.

    Args:
        image: PIL Image or numpy ndarray.
        model: Optional pre-loaded model. If None, loaded via load_model.
        processor: Optional pre-loaded processor. If None, loaded via load_model.
        device: Optional torch device. If None, auto-detected.

    Returns:
        dict with keys:
            - 'predicted_class' (str): Name of predicted class.
            - 'class_id' (int): Numerical class index.
            - 'confidence' (float): Softmax probability of top class.
            - 'top5' (list[tuple[str, float]]): Top 5 classes and probabilities.
            - 'logits' (torch.Tensor): 1D tensor of logits on CPU.
            - 'probabilities' (torch.Tensor): 1D tensor of softmax probabilities on CPU.
            - 'attentions' (list[torch.Tensor]): List of attention tensors (one per layer),
              each of shape [1, num_heads, seq_len, seq_len] on CPU.
    """
    results = predict_batch([image], model=model, processor=processor, device=device)
    return results[0]
