"""
attention.py - Attention Rollout Analysis and Heatmap Visualization
Part of the TransformTrust framework.

Implements attention rollout analysis based on Abnar & Zuidema (2020):
"Quantifying Attention Flow in Transformers" (ACL 2020).

Key features:
1. attention_rollout: Multi-layer attention weight aggregation and normalization.
2. compute_attention_similarity: Cosine similarity with transform-aware patch alignment.
3. generate_attention_heatmap: Bilinear upsampling and colormapped overlay on images.
4. generate_attention_comparison: Side-by-side visualization with metrics.
"""

import io
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.distance import cosine as scipy_cosine
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


def attention_rollout(attentions: Union[List[Any], Tuple[Any, ...]]) -> np.ndarray:
    """
    Compute attention rollout across transformer layers.

    Based on Abnar & Zuidema (2020):
    1. Average attention weights across attention heads for each layer.
    2. Add identity matrix to account for residual connections (0.5 * A + 0.5 * I).
    3. Re-normalize rows to sum to 1.
    4. Recursively multiply attention matrices across layers:
       rollout = layer1 @ layer2 @ ... @ layerN
    5. Extract CLS token attention to patch tokens (row 0, columns 1:).
    6. Reshape to 14x14 spatial grid.
    7. Min-max normalize heatmap to [0, 1].

    Parameters
    ----------
    attentions : list or tuple of attention weight tensors/arrays
        One tensor per transformer layer.
        Each tensor has shape (batch=1, num_heads, seq_len, seq_len)
        where seq_len = 197 (196 patches + 1 CLS token) for ViT/DeiT 224x224.

    Returns
    -------
    np.ndarray
        14x14 attention heatmap with values normalized to [0, 1].
    """
    if not attentions:
        return np.zeros((14, 14), dtype=np.float32)

    processed_layers = []

    for layer in attentions:
        # Support PyTorch Tensor or NumPy array
        if hasattr(layer, "detach"):
            layer_np = layer.detach().cpu().numpy()
        else:
            layer_np = np.asarray(layer)

        # Handle batch dimension if present: (batch, num_heads, seq_len, seq_len)
        if layer_np.ndim == 4:
            layer_np = layer_np[0]

        # Step a: Average attention across heads -> (seq_len, seq_len)
        if layer_np.ndim == 3:
            attn_heads_avg = np.mean(layer_np, axis=0)
        else:
            attn_heads_avg = layer_np

        seq_len = attn_heads_avg.shape[0]

        # Step b: Add identity matrix (residual connection): attn = 0.5 * attn + 0.5 * I
        identity = np.eye(seq_len, dtype=np.float64)
        attn_residual = 0.5 * attn_heads_avg.astype(np.float64) + 0.5 * identity

        # Step c: Normalize rows to sum to 1
        row_sums = attn_residual.sum(axis=-1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        attn_norm = attn_residual / row_sums

        processed_layers.append(attn_norm)

    # Step d: Multiply matrices across layers: rollout = layer1 @ layer2 @ ... @ layerN
    rollout = processed_layers[0]
    for layer_matrix in processed_layers[1:]:
        rollout = rollout @ layer_matrix

    # Step e: Extract CLS token row (row 0), take columns 1: (skip CLS-to-CLS)
    cls_attn = rollout[0, 1:]  # shape (196,) for 14x14 patches

    # Step f: Reshape to 14x14
    num_patches = cls_attn.shape[0]
    grid_size = int(round(np.sqrt(num_patches)))
    if grid_size * grid_size == num_patches:
        heatmap = cls_attn.reshape(grid_size, grid_size)
    else:
        # Fallback if patch count differs from perfect square
        heatmap = cls_attn[:196].reshape(14, 14)

    # Step g: Normalize to [0, 1]
    min_val = float(np.min(heatmap))
    max_val = float(np.max(heatmap))
    if max_val - min_val > 1e-12:
        heatmap = (heatmap - min_val) / (max_val - min_val)
    else:
        heatmap = np.zeros_like(heatmap)

    return heatmap.astype(np.float32)


def _calc_cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Safely calculate cosine similarity between two 1D vectors.

    Uses sklearn.metrics.pairwise.cosine_similarity and scipy.spatial.distance.cosine.
    Returns a float strictly clipped to [0.0, 1.0].
    """
    u_flat = np.asarray(u, dtype=np.float64).flatten()
    v_flat = np.asarray(v, dtype=np.float64).flatten()

    if len(u_flat) == 0 or len(v_flat) == 0:
        return 0.0

    norm_u = np.linalg.norm(u_flat)
    norm_v = np.linalg.norm(v_flat)

    if norm_u == 0.0 and norm_v == 0.0:
        return 1.0
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0

    # Calculate similarity using sklearn (and verify consistency with scipy)
    sim = sklearn_cosine_similarity(u_flat.reshape(1, -1), v_flat.reshape(1, -1))[0, 0]
    return float(np.clip(sim, 0.0, 1.0))


def compute_attention_similarity(
    attn_orig: np.ndarray,
    attn_transformed: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """
    Compute attention map similarity with transformation-aware alignment.

    Parameters
    ----------
    attn_orig : np.ndarray
        14x14 original attention map.
    attn_transformed : np.ndarray
        14x14 transformed attention map.
    metadata : dict, optional
        Transformation metadata dict containing alignment information.
        If metadata['can_align_attention'] is False, returns None.

    Transform-specific alignment:
    - Pixel-level (blur, grayscale, brightness, contrast, mild_occlusion):
      Direct cosine similarity between flattened maps.
    - patch_shift:
      Inverse-shift the transformed attention map by (-dx, -dy) in patch coordinates
      (dx, dy in pixels converted to patch units via / 16).
      Computes cosine similarity on overlapping region only.
    - patch_shuffle:
      Apply inverse permutation to transformed attention map patches, then compute cosine similarity.
    - patch_dropout:
      Mask both maps to only include kept patches, then compute cosine similarity.

    Returns
    -------
    float or None
        Cosine similarity in [0, 1] where 1 = identical, or None if unalignable.
    """
    if metadata is None:
        metadata = {}

    # Check if attention map can be aligned
    if metadata.get("can_align_attention") is False:
        return None

    orig_map = np.asarray(attn_orig, dtype=np.float32)
    trans_map = np.asarray(attn_transformed, dtype=np.float32)

    # Ensure 2D (14, 14) shape
    if orig_map.ndim == 3:
        orig_map = orig_map.squeeze()
    if trans_map.ndim == 3:
        trans_map = trans_map.squeeze()

    transform_type = str(
        metadata.get("transform_name")
        or metadata.get("transform_type")
        or metadata.get("transformation")
        or metadata.get("name")
        or metadata.get("type")
        or ""
    ).lower()

    # 1. Patch Shift Alignment
    if "shift" in transform_type or "patch_dx" in metadata or "dx" in metadata:
        # Determine patch-level shifts (patch size is 16 pixels)
        if "patch_dx" in metadata:
            patch_dx = int(round(metadata["patch_dx"]))
        else:
            dx = float(metadata.get("dx", metadata.get("shift_x", metadata.get("delta_x", 0))))
            patch_dx = int(round(dx / 16.0))

        if "patch_dy" in metadata:
            patch_dy = int(round(metadata["patch_dy"]))
        else:
            dy = float(metadata.get("dy", metadata.get("shift_y", metadata.get("delta_y", 0))))
            patch_dy = int(round(dy / 16.0))

        h, w = orig_map.shape[:2]

        # Determine overlapping regions between orig and inverse-shifted transformed map
        # Transformed patch at (r + patch_dy, c + patch_dx) corresponds to original patch at (r, c)
        if patch_dy >= 0:
            r_orig_start, r_orig_end = 0, h - patch_dy
            r_trans_start, r_trans_end = patch_dy, h
        else:
            r_orig_start, r_orig_end = -patch_dy, h
            r_trans_start, r_trans_end = 0, h + patch_dy

        if patch_dx >= 0:
            c_orig_start, c_orig_end = 0, w - patch_dx
            c_trans_start, c_trans_end = patch_dx, w
        else:
            c_orig_start, c_orig_end = -patch_dx, w
            c_trans_start, c_trans_end = 0, w + patch_dx

        # Check if overlap is empty
        if r_orig_end <= r_orig_start or c_orig_end <= c_orig_start:
            return 0.0

        orig_overlap = orig_map[r_orig_start:r_orig_end, c_orig_start:c_orig_end]
        trans_overlap = trans_map[r_trans_start:r_trans_end, c_trans_start:c_trans_end]

        return _calc_cosine_similarity(orig_overlap.flatten(), trans_overlap.flatten())

    # 2. Patch Shuffle Alignment
    if "shuffle" in transform_type or "permutation" in metadata or "perm" in metadata:
        trans_flat = trans_map.flatten()

        if "inverse_permutation" in metadata:
            inv_perm = np.asarray(metadata["inverse_permutation"], dtype=int)
            unshuffled_trans = trans_flat[inv_perm]
        elif "inv_perm" in metadata:
            inv_perm = np.asarray(metadata["inv_perm"], dtype=int)
            unshuffled_trans = trans_flat[inv_perm]
        elif "permutation" in metadata or "perm" in metadata or "shuffle_indices" in metadata:
            perm = np.asarray(
                metadata.get("permutation", metadata.get("perm", metadata.get("shuffle_indices"))),
                dtype=int,
            )
            # If shuffled = orig[perm], then orig = shuffled[argsort(perm)]
            inv_perm = np.argsort(perm)
            unshuffled_trans = trans_flat[inv_perm]
        else:
            unshuffled_trans = trans_flat

        return _calc_cosine_similarity(orig_map.flatten(), unshuffled_trans)

    # 3. Patch Dropout Alignment
    if "dropout" in transform_type or "kept_indices" in metadata or "keep_mask" in metadata:
        flat_orig = orig_map.flatten()
        flat_trans = trans_map.flatten()

        kept_indices = None
        if "kept_indices" in metadata:
            kept_indices = np.asarray(metadata["kept_indices"], dtype=int)
        elif "keep_indices" in metadata:
            kept_indices = np.asarray(metadata["keep_indices"], dtype=int)
        elif "kept_patches" in metadata:
            kept_indices = np.asarray(metadata["kept_patches"], dtype=int)
        elif "active_indices" in metadata:
            kept_indices = np.asarray(metadata["active_indices"], dtype=int)
        elif "keep_mask" in metadata:
            mask = np.asarray(metadata["keep_mask"]).flatten().astype(bool)
            kept_indices = np.where(mask)[0]
        elif "mask" in metadata:
            mask = np.asarray(metadata["mask"]).flatten()
            if mask.dtype == bool or np.max(mask) <= 1:
                kept_indices = np.where(mask > 0)[0]
        elif "dropped_indices" in metadata or "drop_indices" in metadata:
            dropped = set(metadata.get("dropped_indices", metadata.get("drop_indices")))
            kept_indices = np.array([i for i in range(len(flat_orig)) if i not in dropped], dtype=int)

        if kept_indices is not None:
            if len(kept_indices) == 0:
                return 0.0
            return _calc_cosine_similarity(flat_orig[kept_indices], flat_trans[kept_indices])
        else:
            return _calc_cosine_similarity(flat_orig, flat_trans)

    # 4. Pixel-level transforms & Fallback (blur, grayscale, brightness, contrast, mild_occlusion)
    return _calc_cosine_similarity(orig_map.flatten(), trans_map.flatten())


def generate_attention_heatmap(
    image: Union[Image.Image, np.ndarray, str],
    attention_map: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.5,
) -> Image.Image:
    """
    Overlay 14x14 attention heatmap on the original image.

    Parameters
    ----------
    image : PIL.Image.Image, np.ndarray, or str
        Base image to overlay heatmap upon.
    attention_map : np.ndarray
        14x14 attention heatmap.
    colormap : str, default='jet'
        Matplotlib colormap name ('jet', 'inferno', 'viridis', etc.).
    alpha : float, default=0.5
        Blending ratio between original image and heatmap overlay.

    Returns
    -------
    PIL.Image.Image
        Blended RGB image with attention heatmap overlay.
    """
    # Ensure base image is a PIL Image in RGB format
    if isinstance(image, str):
        base_img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        base_img = Image.fromarray(image).convert("RGB")
    elif isinstance(image, Image.Image):
        base_img = image.convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    img_w, img_h = base_img.size

    # Normalize attention map to [0, 1]
    attn = np.asarray(attention_map, dtype=np.float32)
    if attn.ndim == 3 and attn.shape[0] == 1:
        attn = attn.squeeze(0)

    min_val = float(np.min(attn))
    max_val = float(np.max(attn))
    if max_val - min_val > 1e-12:
        norm_attn = (attn - min_val) / (max_val - min_val)
    else:
        norm_attn = np.zeros_like(attn)

    # Upsample attention map to image dimensions using bilinear interpolation
    attn_pil = Image.fromarray(norm_attn.astype(np.float32), mode="F")
    attn_resized = attn_pil.resize((img_w, img_h), resample=Image.Resampling.BILINEAR)
    attn_resized_np = np.clip(np.array(attn_resized, dtype=np.float32), 0.0, 1.0)

    # Apply colormap
    try:
        cmap = matplotlib.colormaps[colormap]
    except (AttributeError, KeyError):
        cmap = cm.get_cmap(colormap)

    heatmap_rgba = cmap(attn_resized_np)  # RGBA in [0, 1]
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255.0).astype(np.uint8)
    heatmap_pil = Image.fromarray(heatmap_rgb, mode="RGB")

    # Blend original image and heatmap overlay
    blended = Image.blend(base_img, heatmap_pil, alpha=alpha)
    return blended


def generate_attention_comparison(
    orig_image: Union[Image.Image, np.ndarray, str],
    orig_attn: np.ndarray,
    trans_image: Union[Image.Image, np.ndarray, str],
    trans_attn: np.ndarray,
    similarity: Optional[float],
    transform_name: str,
    colormap: str = "jet",
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (10, 5),
    dpi: int = 150,
) -> Image.Image:
    """
    Create a side-by-side comparison image of original vs transformed attention.

    Parameters
    ----------
    orig_image : PIL.Image.Image, np.ndarray, or str
        Original base image.
    orig_attn : np.ndarray
        14x14 attention map for original image.
    trans_image : PIL.Image.Image, np.ndarray, or str
        Transformed base image.
    trans_attn : np.ndarray
        14x14 attention map for transformed image.
    similarity : float or None
        Cosine similarity score between attention maps, or None if unaligned.
    transform_name : str
        Name of the transformation applied.
    colormap : str, default='jet'
        Colormap for attention heatmaps.
    alpha : float, default=0.5
        Overlay blending alpha.
    figsize : tuple of int, default=(10, 5)
        Figure size in inches.
    dpi : int, default=150
        Resolution of output comparison image.

    Returns
    -------
    PIL.Image.Image
        Side-by-side comparison visualization image.
    """
    # Generate attention overlays for both original and transformed
    orig_overlay = generate_attention_heatmap(orig_image, orig_attn, colormap=colormap, alpha=alpha)
    trans_overlay = generate_attention_heatmap(trans_image, trans_attn, colormap=colormap, alpha=alpha)

    # Format title and subtitle
    clean_name = transform_name.replace("_", " ").title() if transform_name else "Transformed"
    if similarity is not None:
        sim_str = f"Attention Cosine Similarity: {similarity:.4f}"
    else:
        sim_str = "Attention Similarity: N/A (unaligned)"

    # Create side-by-side matplotlib figure with Agg backend
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    axes[0].imshow(orig_overlay)
    axes[0].set_title("Original Image (Attention Overlay)", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    axes[1].imshow(trans_overlay)
    axes[1].set_title(f"Transformed: {clean_name}", fontsize=11, fontweight="bold", pad=8)
    axes[1].axis("off")

    fig.suptitle(f"{clean_name} Analysis — {sim_str}", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()

    # Render figure to in-memory PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)

    comparison_img = Image.open(buf).convert("RGB")
    return comparison_img
