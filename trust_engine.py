"""
TransformTrust: Trust Score and Fragility Engine
=================================================
Computes the TransformTrust reliability score, fragility analysis,
and confidence-trust mismatch detection.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def compute_trust_score(
    original_prediction: Dict[str, Any],
    transform_results: Dict[str, Dict[str, Any]],
    mild_transform_names: List[str],
) -> Dict[str, Any]:
    """Compute the TransformTrust reliability score.

    Trust Score = 0.5 * Agreement Rate + 0.5 * Mean Attention Similarity
    Scaled to 0-100.

    Args:
        original_prediction: Dict with 'predicted_class', 'confidence', etc.
        transform_results: Dict mapping transform_name -> {
            'prediction': predict result,
            'attention_map': 14x14 np.ndarray,
            'attention_similarity': float or None,
            'metadata': transform metadata
        }
        mild_transform_names: List of names that are 'mild' transforms.

    Returns:
        Dict with trust_score, agreement_rate, mean_attention_similarity, etc.
    """
    orig_class = original_prediction["predicted_class"]

    # Count agreements among mild transforms
    num_agreeing = 0
    num_total_mild = 0
    valid_sims = []

    for tname in mild_transform_names:
        if tname not in transform_results:
            continue
        tres = transform_results[tname]
        num_total_mild += 1

        if tres["prediction"]["predicted_class"] == orig_class:
            num_agreeing += 1

        sim = tres.get("attention_similarity")
        if sim is not None:
            valid_sims.append(sim)

    # Agreement rate
    agreement_rate = num_agreeing / max(num_total_mild, 1)

    # Mean attention similarity (only from mild transforms with valid comparisons)
    if valid_sims:
        mean_attn_sim = float(np.mean(valid_sims))
    else:
        mean_attn_sim = agreement_rate  # Fallback: use agreement_rate if no valid sims

    # Trust score
    trust_score = (0.5 * agreement_rate + 0.5 * mean_attn_sim) * 100.0

    return {
        "trust_score": trust_score,
        "agreement_rate": agreement_rate,
        "mean_attention_similarity": mean_attn_sim,
        "original_confidence": original_prediction["confidence"],
        "num_agreeing_mild": num_agreeing,
        "num_total_mild": num_total_mild,
        "num_valid_attention_comparisons": len(valid_sims),
    }


def compute_fragility(
    transform_name: str,
    prediction_agrees: bool,
    attention_similarity: Optional[float],
) -> float:
    """Compute fragility for a single transformation.

    Fragility = 1 - (prediction_agreement * attention_similarity)

    Args:
        transform_name: Name of the transformation.
        prediction_agrees: Whether the predicted class matches the original.
        attention_similarity: Cosine similarity of attention maps, or None.

    Returns:
        Fragility score in [0, 1].
    """
    pred_agree_val = 1.0 if prediction_agrees else 0.0

    if attention_similarity is not None:
        fragility = 1.0 - (pred_agree_val * attention_similarity)
    else:
        # If attention similarity is unavailable, fragility is based only on prediction agreement
        fragility = 1.0 - pred_agree_val

    return float(np.clip(fragility, 0.0, 1.0))


def compute_all_fragilities(
    original_prediction: Dict[str, Any],
    transform_results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute fragility for all transformations.

    Args:
        original_prediction: Original prediction result dict.
        transform_results: Dict of transform results.

    Returns:
        List of dicts sorted by fragility (most fragile first), each containing:
        transform_name, fragility, prediction_agrees, attention_similarity,
        predicted_class, confidence, type.
    """
    orig_class = original_prediction["predicted_class"]
    results = []

    for tname, tres in transform_results.items():
        pred = tres["prediction"]
        agrees = pred["predicted_class"] == orig_class
        sim = tres.get("attention_similarity")
        frag = compute_fragility(tname, agrees, sim)
        t_type = tres.get("metadata", {}).get("type", "unknown")

        results.append({
            "transform_name": tname,
            "fragility": frag,
            "prediction_agrees": agrees,
            "attention_similarity": sim,
            "predicted_class": pred["predicted_class"],
            "confidence": pred["confidence"],
            "type": t_type,
        })

    # Sort by fragility descending
    results.sort(key=lambda x: x["fragility"], reverse=True)
    return results


def detect_confidence_trust_mismatch(
    confidence: float,
    trust_score: float,
    threshold: float = 20.0,
) -> Dict[str, Any]:
    """Detect mismatch between model confidence and trust score.

    Args:
        confidence: Model confidence (0-100 scale).
        trust_score: TransformTrust score (0-100 scale).
        threshold: Gap threshold for triggering a warning.

    Returns:
        Dict with is_mismatch, confidence, trust_score, mismatch_gap,
        severity, and explanation.
    """
    gap = confidence - trust_score
    is_mismatch = gap > threshold

    if gap <= threshold:
        severity = "none"
    elif gap <= 30:
        severity = "mild"
    elif gap <= 50:
        severity = "moderate"
    else:
        severity = "severe"

    explanations = {
        "none": (
            "The model's confidence and the TransformTrust reliability score are "
            "reasonably aligned. No significant concern detected."
        ),
        "mild": (
            "The model reports somewhat higher confidence than what transformation-based "
            "consistency suggests. This is a mild signal — the prediction may still be "
            "correct but could benefit from a quick visual check."
        ),
        "moderate": (
            "There is a notable gap between the model's confidence and transformation-based "
            "reliability. The prediction is not necessarily wrong, but this discrepancy "
            "warrants closer inspection. The model may be over-confident relative to how "
            "stable its prediction is under perturbations."
        ),
        "severe": (
            "The model expresses very high confidence but transformation-based analysis "
            "reveals significant instability in predictions and/or attention patterns. "
            "This does NOT prove the prediction is incorrect, but strongly suggests the "
            "prediction should be verified through additional means before relying on it."
        ),
    }

    return {
        "is_mismatch": is_mismatch,
        "confidence": confidence,
        "trust_score": trust_score,
        "mismatch_gap": gap,
        "severity": severity,
        "explanation": explanations.get(severity, ""),
    }


def run_full_analysis(
    image,
    predict_fn: Callable,
    attention_rollout_fn: Callable,
    attention_similarity_fn: Callable,
    all_transforms: Dict[str, Tuple],
    mild_transform_names: List[str],
    mismatch_threshold: float = 20.0,
) -> Dict[str, Any]:
    """Run the full TransformTrust analysis pipeline on a single image.

    Args:
        image: PIL Image.
        predict_fn: Function that takes an image and returns prediction dict.
        attention_rollout_fn: Function that takes attentions list and returns 14x14 map.
        attention_similarity_fn: Function(orig_attn, trans_attn, metadata) -> float or None.
        all_transforms: Dict mapping name -> (transform_fn, description).
        mild_transform_names: List of mild transform names.
        mismatch_threshold: Threshold for confidence-trust mismatch.

    Returns:
        Comprehensive results dict.
    """
    # 1. Original prediction
    orig_result = predict_fn(image)
    orig_attn_map = attention_rollout_fn(orig_result["attentions"])

    # 2. Apply transforms and analyze
    transform_results = {}
    for tname, (tfunc, tdesc) in all_transforms.items():
        t_image, t_meta = tfunc(image)
        t_result = predict_fn(t_image)
        t_attn_map = attention_rollout_fn(t_result["attentions"])
        t_sim = attention_similarity_fn(orig_attn_map, t_attn_map, t_meta)

        transform_results[tname] = {
            "prediction": t_result,
            "transformed_image": t_image,
            "attention_map": t_attn_map,
            "attention_similarity": t_sim,
            "metadata": t_meta,
            "description": tdesc,
        }

    # 3. Trust score
    trust_data = compute_trust_score(orig_result, transform_results, mild_transform_names)

    # 4. Fragilities
    fragilities = compute_all_fragilities(orig_result, transform_results)

    # 5. Mismatch detection
    mismatch = detect_confidence_trust_mismatch(
        orig_result["confidence"] * 100,
        trust_data["trust_score"],
        mismatch_threshold,
    )

    return {
        "original_prediction": orig_result,
        "original_attention_map": orig_attn_map,
        "transform_results": transform_results,
        "trust_data": trust_data,
        "fragilities": fragilities,
        "mismatch": mismatch,
    }
