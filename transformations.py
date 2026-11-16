"""
TransformTrust - Image Transformations Module
==============================================
Implements label-preserving mild transformations and structural stress tests
for vision transformer reliability and attention alignment analysis.

Author: Senior ML Engineer
Project: TransformTrust
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

# Default reproducibility seed
DEFAULT_SEED: int = 42

# Resampling filter compatibility (supports Pillow 9.x+ and earlier)
RESAMPLE_FILTER = getattr(Image, "Resampling", Image).BILINEAR


def _ensure_rgb(image: Image.Image) -> Image.Image:
    """Ensure image is in 3-channel RGB mode."""
    if image.mode != "RGB":
        return image.convert("RGB")
    return image.copy()


def _ensure_224(image: Image.Image) -> Image.Image:
    """Ensure image is 224x224 RGB for patch-level operations."""
    rgb_img = _ensure_rgb(image)
    if rgb_img.size != (224, 224):
        return rgb_img.resize((224, 224), resample=RESAMPLE_FILTER)
    return rgb_img


# ============================================================================
# MILD TRANSFORMATIONS (Label-preserving)
# ============================================================================

def gaussian_blur(
    image: Image.Image, radius: float = 2.0
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Apply Gaussian blur to the input image.

    Args:
        image: Input PIL Image.
        radius: Blur radius in pixels (default: 2.0).

    Returns:
        Tuple of (transformed_image, metadata_dict).
    """
    rgb_img = _ensure_rgb(image)
    blurred = rgb_img.filter(ImageFilter.GaussianBlur(radius=radius))

    metadata: Dict[str, Any] = {
        "type": "mild",
        "name": "gaussian_blur",
        "description": f"Gaussian blur with radius={radius}",
        "can_align_attention": True,
        "radius": radius,
        "alignment_info": {
            "type": "none",
            "description": "Pixel-level transform; spatial grid and attention map stay aligned.",
            "radius": radius,
        },
    }
    return blurred, metadata


def grayscale(
    image: Image.Image
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Convert image to grayscale, then back to 3-channel RGB.

    Args:
        image: Input PIL Image.

    Returns:
        Tuple of (transformed_image, metadata_dict).
    """
    rgb_img = _ensure_rgb(image)
    gray_3ch = rgb_img.convert("L").convert("RGB")

    metadata: Dict[str, Any] = {
        "type": "mild",
        "name": "grayscale",
        "description": "Grayscale conversion restored to 3-channel RGB",
        "can_align_attention": True,
        "alignment_info": {
            "type": "none",
            "description": "Pixel-level transform; spatial grid and attention map stay aligned.",
        },
    }
    return gray_3ch, metadata


def brightness(
    image: Image.Image, factor: float = 1.5
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Increase image brightness by the specified factor.

    Args:
        image: Input PIL Image.
        factor: Brightness multiplier (default: 1.5).

    Returns:
        Tuple of (transformed_image, metadata_dict).
    """
    rgb_img = _ensure_rgb(image)
    enhancer = ImageEnhance.Brightness(rgb_img)
    brightened = enhancer.enhance(factor)

    metadata: Dict[str, Any] = {
        "type": "mild",
        "name": "brightness",
        "description": f"Brightness adjustment by factor {factor}",
        "can_align_attention": True,
        "factor": factor,
        "alignment_info": {
            "type": "none",
            "description": "Pixel-level transform; spatial grid and attention map stay aligned.",
            "factor": factor,
        },
    }
    return brightened, metadata


def contrast(
    image: Image.Image, factor: float = 1.5
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Increase image contrast by the specified factor.

    Args:
        image: Input PIL Image.
        factor: Contrast multiplier (default: 1.5).

    Returns:
        Tuple of (transformed_image, metadata_dict).
    """
    rgb_img = _ensure_rgb(image)
    enhancer = ImageEnhance.Contrast(rgb_img)
    contrasted = enhancer.enhance(factor)

    metadata: Dict[str, Any] = {
        "type": "mild",
        "name": "contrast",
        "description": f"Contrast adjustment by factor {factor}",
        "can_align_attention": True,
        "factor": factor,
        "alignment_info": {
            "type": "none",
            "description": "Pixel-level transform; spatial grid and attention map stay aligned.",
            "factor": factor,
        },
    }
    return contrasted, metadata


def mild_occlusion(
    image: Image.Image, coverage: float = 0.10, seed: int = DEFAULT_SEED
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Add a small solid gray rectangle covering ~10% of the image at a seeded random position.

    Args:
        image: Input PIL Image.
        coverage: Fractional image area to occlude (default: 0.10 ~ 10%).
        seed: Random seed for positioning (default: 42).

    Returns:
        Tuple of (transformed_image, metadata_dict).
    """
    rgb_img = _ensure_rgb(image)
    w, h = rgb_img.size

    # Compute rectangle dimensions matching aspect ratio to achieve target area coverage
    side_ratio = math.sqrt(coverage)
    rect_w = max(1, min(w, int(round(w * side_ratio))))
    rect_h = max(1, min(h, int(round(h * side_ratio))))

    rng = np.random.default_rng(seed)
    max_x = max(0, w - rect_w)
    max_y = max(0, h - rect_h)
    x0 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
    y0 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    x1 = min(w, x0 + rect_w)
    y1 = min(h, y0 + rect_h)

    occluded = rgb_img.copy()
    draw = ImageDraw.Draw(occluded)
    fill_color = (128, 128, 128)
    draw.rectangle([x0, y0, x1, y1], fill=fill_color)

    metadata: Dict[str, Any] = {
        "type": "mild",
        "name": "mild_occlusion",
        "description": f"Mild gray occlusion covering ~{int(coverage * 100)}% of the image",
        "can_align_attention": True,
        "bbox": (x0, y0, x1, y1),
        "alignment_info": {
            "type": "occlusion",
            "description": "Spatial coordinate system preserved; gray patch occluded at bbox.",
            "bbox": (x0, y0, x1, y1),
            "coverage": coverage,
            "seed": seed,
            "fill_color": fill_color,
        },
    }
    return occluded, metadata


# ============================================================================
# STRUCTURAL STRESS TESTS
# ============================================================================

def patch_shift(
    image: Image.Image, dx: int = 8, dy: int = 8
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Shift image by dx, dy pixels (default 8, 8) with zero-padding (black, no wrap).
    Resizes image to 224x224 prior to shifting.

    Args:
        image: Input PIL Image.
        dx: Horizontal shift in pixels (positive = right, negative = left).
        dy: Vertical shift in pixels (positive = down, negative = up).

    Returns:
        Tuple of (transformed_image, metadata_dict).
    """
    base_img = _ensure_224(image)
    img_arr = np.array(base_img)
    h, w, c = img_arr.shape

    out_arr = np.zeros_like(img_arr)

    # Calculate valid source and destination slices
    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)
    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)

    dst_y1 = max(0, dy)
    dst_y2 = min(h, h + dy)
    dst_x1 = max(0, dx)
    dst_x2 = min(w, w + dx)

    if src_y2 > src_y1 and src_x2 > src_x1 and dst_y2 > dst_y1 and dst_x2 > dst_x1:
        out_arr[dst_y1:dst_y2, dst_x1:dst_x2] = img_arr[src_y1:src_y2, src_x1:src_x2]

    shifted = Image.fromarray(out_arr)

    metadata: Dict[str, Any] = {
        "type": "stress",
        "name": "patch_shift",
        "description": f"Shift image by ({dx}, {dy}) pixels with black padding (no wrap)",
        "can_align_attention": True,
        "dx": dx,
        "dy": dy,
        "alignment_info": {
            "type": "shift",
            "description": "Image shifted by (dx, dy); attention map requires corresponding shift.",
            "dx": dx,
            "dy": dy,
            "grid_size": (14, 14),
            "patch_size": 16,
            "image_size": (224, 224),
        },
    }
    return shifted, metadata


def patch_shuffle(
    image: Image.Image, seed: int = DEFAULT_SEED
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Divide 224x224 image into a 14x14 grid of 16x16 patches and shuffle them
    using a fixed random seed. Returns the permutation array for inverse alignment.

    Args:
        image: Input PIL Image.
        seed: Random seed for shuffling (default: 42).

    Returns:
        Tuple of (transformed_image, metadata_dict).
    """
    base_img = _ensure_224(image)
    img_arr = np.array(base_img)

    num_rows, num_cols = 14, 14
    patch_size = 16
    total_patches = num_rows * num_cols  # 196

    # Extract all patches in row-major order: index = r * 14 + c
    patches = []
    for r in range(num_rows):
        for c in range(num_cols):
            y1 = r * patch_size
            y2 = y1 + patch_size
            x1 = c * patch_size
            x2 = x1 + patch_size
            patches.append(img_arr[y1:y2, x1:x2].copy())

    # Generate deterministic permutation
    rng = np.random.default_rng(seed)
    # perm[dst_idx] = src_idx: patch at dst_idx in transformed image comes from src_idx of original image
    permutation = rng.permutation(total_patches)
    # inv_perm[src_idx] = dst_idx: inverse mapping to restore original patch order
    inverse_permutation = np.argsort(permutation)

    # Assemble shuffled image
    shuffled_arr = np.zeros_like(img_arr)
    for dst_idx, src_idx in enumerate(permutation):
        r_dst = dst_idx // num_cols
        c_dst = dst_idx % num_cols
        y1 = r_dst * patch_size
        y2 = y1 + patch_size
        x1 = c_dst * patch_size
        x2 = x1 + patch_size
        shuffled_arr[y1:y2, x1:x2] = patches[src_idx]

    shuffled_img = Image.fromarray(shuffled_arr)

    metadata: Dict[str, Any] = {
        "type": "stress",
        "name": "patch_shuffle",
        "description": "Shuffle 14x14 grid of 16x16 patches with fixed seed",
        "can_align_attention": True,
        "permutation": permutation,
        "inverse_permutation": inverse_permutation,
        "alignment_info": {
            "type": "shuffle",
            "description": "Patches shuffled by permutation array; inverse_permutation restores order.",
            "permutation": permutation,
            "inverse_permutation": inverse_permutation,
            "grid_size": (num_rows, num_cols),
            "patch_size": patch_size,
            "num_patches": total_patches,
            "seed": seed,
        },
    }
    return shuffled_img, metadata


def patch_dropout(
    image: Image.Image, drop_rate: float = 0.25, seed: int = DEFAULT_SEED
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Randomly zero out 25% of 16x16 patches (replace with neutral gray: 128, 128, 128).
    Returns binary mask of kept patches (shape 196, True=kept, False=dropped).

    Args:
        image: Input PIL Image.
        drop_rate: Fraction of patches to drop (default: 0.25 ~ 49 patches).
        seed: Random seed for selecting dropped patches (default: 42).

    Returns:
        Tuple of (transformed_image, metadata_dict).
    """
    base_img = _ensure_224(image)
    img_arr = np.array(base_img)

    num_rows, num_cols = 14, 14
    patch_size = 16
    total_patches = num_rows * num_cols  # 196
    num_drop = int(round(total_patches * drop_rate))

    rng = np.random.default_rng(seed)
    drop_indices = rng.choice(total_patches, size=num_drop, replace=False)

    # Boolean mask: True = kept, False = dropped
    kept_mask = np.ones(total_patches, dtype=bool)
    kept_mask[drop_indices] = False

    dropout_arr = img_arr.copy()
    gray_patch = np.full((patch_size, patch_size, 3), 128, dtype=np.uint8)

    for idx in drop_indices:
        r = idx // num_cols
        c = idx % num_cols
        y1 = r * patch_size
        y2 = y1 + patch_size
        x1 = c * patch_size
        x2 = x1 + patch_size
        dropout_arr[y1:y2, x1:x2] = gray_patch

    dropout_img = Image.fromarray(dropout_arr)

    metadata: Dict[str, Any] = {
        "type": "stress",
        "name": "patch_dropout",
        "description": f"Randomly drop {int(drop_rate * 100)}% of 16x16 patches (neutral gray replacement)",
        "can_align_attention": True,
        "kept_mask": kept_mask,
        "alignment_info": {
            "type": "dropout",
            "description": "Dropped patches replaced with gray; evaluate attention on kept_mask only.",
            "kept_mask": kept_mask,
            "drop_rate": drop_rate,
            "num_dropped": num_drop,
            "grid_size": (num_rows, num_cols),
            "patch_size": patch_size,
            "num_patches": total_patches,
            "seed": seed,
            "fill_value": (128, 128, 128),
        },
    }
    return dropout_img, metadata


# ============================================================================
# ATTENTION ALIGNMENT UTILITIES
# ============================================================================

def align_patch_attention(
    attention_map: np.ndarray,
    metadata: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Inverse-align transformed patch-level attention back to original patch coordinates.

    Args:
        attention_map: Transformed attention map. Can be shape (196,) for 1D patch vector
                       or (14, 14) for 2D spatial grid.
        metadata: Metadata dict returned by the transformation function.

    Returns:
        Tuple of (aligned_attention, valid_mask):
            aligned_attention: Attention aligned to the original 14x14 grid.
            valid_mask: Optional boolean mask (shape 196 or 14x14) indicating which patches
                        are valid for comparison (e.g., False for dropped patches).
    """
    align_info = metadata.get("alignment_info", {})
    trans_type = align_info.get("type", "none")

    is_2d = attention_map.ndim == 2 and attention_map.shape == (14, 14)
    flat_attn = attention_map.flatten()

    if trans_type == "shuffle":
        # Permutation: shuffled_patch[dst_idx] = original_patch[src_idx]
        # To restore original order: aligned[src_idx] = shuffled[dst_idx]
        inv_perm = align_info["inverse_permutation"]
        # inv_perm maps src_idx -> dst_idx, so flat_attn[inv_perm] retrieves original order:
        aligned_flat = flat_attn[inv_perm]
        aligned = aligned_flat.reshape((14, 14)) if is_2d else aligned_flat
        return aligned, None

    elif trans_type == "dropout":
        # Spatial positions stay identical, but dropped patches should be masked
        kept_mask = align_info["kept_mask"]
        valid_mask = kept_mask.reshape((14, 14)) if is_2d else kept_mask
        aligned = attention_map.copy()
        return aligned, valid_mask

    elif trans_type == "shift":
        # Pixel-level shift by (dx, dy).
        # For a ViT with 16x16 patches, dx=8 is a half-patch shift.
        # Attention map is returned as-is alongside valid region metadata.
        return attention_map.copy(), None

    else:
        # Identity alignment (mild transforms preserve patch coordinate grid)
        return attention_map.copy(), None


# ============================================================================
# REGISTRIES AND DISPATCHER
# ============================================================================

MILD_TRANSFORMS: Dict[str, Tuple[Callable[[Image.Image], Tuple[Image.Image, Dict[str, Any]]], str]] = {
    "gaussian_blur": (gaussian_blur, "Gaussian blur with radius=2 (mild smoothing)"),
    "grayscale": (grayscale, "Grayscale conversion to 3-channel RGB (removes color cues)"),
    "brightness": (brightness, "Increase brightness by 1.5x (illumination variance)"),
    "contrast": (contrast, "Increase contrast by 1.5x (dynamic range stretch)"),
    "mild_occlusion": (mild_occlusion, "Mild gray rectangular occlusion (~10% image area)"),
}

STRESS_TRANSFORMS: Dict[str, Tuple[Callable[[Image.Image], Tuple[Image.Image, Dict[str, Any]]], str]] = {
    "patch_shift": (patch_shift, "Shift image by 8px right & down with zero padding"),
    "patch_shuffle": (patch_shuffle, "Divide into 14x14 grid of 16x16 patches & shuffle (breaks geometry)"),
    "patch_dropout": (patch_dropout, "Randomly drop 25% of 16x16 patches with neutral gray"),
}

ALL_TRANSFORMS: Dict[str, Tuple[Callable[[Image.Image], Tuple[Image.Image, Dict[str, Any]]], str]] = {
    **MILD_TRANSFORMS,
    **STRESS_TRANSFORMS,
}


def apply_all_transforms(
    image: Image.Image
) -> Dict[str, Tuple[Image.Image, Dict[str, Any]]]:
    """
    Apply all mild and stress transformations to the given PIL image.

    Args:
        image: Input PIL Image (any size or mode).

    Returns:
        Dict mapping transform_name -> (transformed_pil_image, metadata_dict).
    """
    results: Dict[str, Tuple[Image.Image, Dict[str, Any]]] = {}
    for name, (fn, _) in ALL_TRANSFORMS.items():
        results[name] = fn(image)
    return results
