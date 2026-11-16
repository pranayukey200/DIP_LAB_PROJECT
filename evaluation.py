"""
TransformTrust: Evaluation Module
===================================
Runs batch evaluation on the Imagenette dataset, computing accuracy, AUROC,
high-trust vs low-trust accuracy, and per-transformation metrics.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_dataset(
    images_with_labels: List[Tuple[Any, str, int]],
    predict_fn: Callable,
    attention_rollout_fn: Callable,
    attention_similarity_fn: Callable,
    transforms_dict: Dict[str, Tuple],
    mild_transform_names: List[str],
    output_dir: str,
    mismatch_threshold: float = 20.0,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Run evaluation on a dataset of labelled images.

    Args:
        images_with_labels: List of (PIL.Image, ground_truth_str, imagenet_idx).
        predict_fn: Prediction function.
        attention_rollout_fn: Attention rollout function.
        attention_similarity_fn: Attention similarity function.
        transforms_dict: All transforms dict.
        mild_transform_names: List of mild transform names.
        output_dir: Directory to save results.
        mismatch_threshold: Confidence-trust mismatch threshold.
        progress_callback: Optional callback(current, total, message).

    Returns:
        Dict with evaluation metrics.
    """
    from trust_engine import compute_trust_score, compute_all_fragilities, detect_confidence_trust_mismatch

    os.makedirs(output_dir, exist_ok=True)
    total = len(images_with_labels)

    per_image_results = []
    all_fragilities_by_transform = {}
    y_true = []  # 1 if prediction is correct
    trust_scores = []

    for i, (image, gt_label, gt_idx) in enumerate(images_with_labels):
        if progress_callback:
            progress_callback(i + 1, total, f"Evaluating image {i+1}/{total}: {gt_label}")

        try:
            # Original prediction
            orig_result = predict_fn(image)
            orig_attn_map = attention_rollout_fn(orig_result["attentions"])

            # Apply transforms
            transform_results = {}
            for tname, (tfunc, tdesc) in transforms_dict.items():
                t_image, t_meta = tfunc(image)
                t_result = predict_fn(t_image)
                t_attn_map = attention_rollout_fn(t_result["attentions"])
                t_sim = attention_similarity_fn(orig_attn_map, t_attn_map, t_meta)

                transform_results[tname] = {
                    "prediction": t_result,
                    "attention_map": t_attn_map,
                    "attention_similarity": t_sim,
                    "metadata": t_meta,
                }

                # Track per-transform fragility
                pred_agrees = t_result["predicted_class"] == orig_result["predicted_class"]
                if tname not in all_fragilities_by_transform:
                    all_fragilities_by_transform[tname] = {
                        "agreements": [],
                        "attention_sims": [],
                        "fragilities": [],
                    }
                all_fragilities_by_transform[tname]["agreements"].append(1 if pred_agrees else 0)
                if t_sim is not None:
                    all_fragilities_by_transform[tname]["attention_sims"].append(t_sim)

                frag_val = 1.0 - ((1.0 if pred_agrees else 0.0) * (t_sim if t_sim is not None else 1.0))
                all_fragilities_by_transform[tname]["fragilities"].append(frag_val)

            # Trust score
            trust_data = compute_trust_score(orig_result, transform_results, mild_transform_names)

            # Check correctness (by ImageNet class ID or label name substring)
            pred_id = orig_result.get("class_id")
            pred_label = orig_result["predicted_class"].lower()
            is_correct = (pred_id == gt_idx) or (gt_label.lower() in pred_label)

            y_true.append(1 if is_correct else 0)
            trust_scores.append(trust_data["trust_score"])

            per_image_results.append({
                "index": i,
                "ground_truth": gt_label,
                "ground_truth_idx": gt_idx,
                "predicted_class": orig_result["predicted_class"],
                "confidence": orig_result["confidence"],
                "trust_score": trust_data["trust_score"],
                "agreement_rate": trust_data["agreement_rate"],
                "attention_similarity": trust_data["mean_attention_similarity"],
                "is_correct": is_correct,
            })

        except Exception as e:
            logger.error(f"Error evaluating image {i}: {e}")
            continue

    # ── Compute aggregate metrics ──
    results = _compute_aggregate_metrics(
        per_image_results, y_true, trust_scores,
        all_fragilities_by_transform, output_dir, mismatch_threshold,
    )

    # Save results
    _save_results(per_image_results, results, all_fragilities_by_transform, output_dir)

    return results


def _compute_aggregate_metrics(
    per_image_results, y_true, trust_scores,
    all_fragilities_by_transform, output_dir, mismatch_threshold,
):
    """Compute aggregate evaluation metrics."""
    from sklearn.metrics import roc_auc_score, confusion_matrix as sk_confusion_matrix

    total = len(per_image_results)
    if total == 0:
        return {"accuracy": 0, "auroc": None, "per_image_results": []}

    num_correct = sum(1 for r in per_image_results if r["is_correct"])
    accuracy = num_correct / total

    # AUROC: use 1 - trust_score/100 as error predictor
    auroc = None
    if len(set(y_true)) > 1:  # Both correct and incorrect exist
        error_scores = [1.0 - ts / 100.0 for ts in trust_scores]
        error_labels = [0 if y == 1 else 1 for y in y_true]  # 1 = error
        try:
            auroc = roc_auc_score(error_labels, error_scores)
        except Exception as e:
            logger.warning(f"Could not compute AUROC: {e}")

    # High-trust vs low-trust accuracy
    high_trust_acc = None
    low_trust_acc = None
    if total >= 4:
        sorted_by_trust = sorted(per_image_results, key=lambda r: r["trust_score"], reverse=True)
        mid = total // 2
        high_trust = sorted_by_trust[:mid]
        low_trust = sorted_by_trust[mid:]

        if high_trust:
            high_trust_acc = sum(1 for r in high_trust if r["is_correct"]) / len(high_trust)
        if low_trust:
            low_trust_acc = sum(1 for r in low_trust if r["is_correct"]) / len(low_trust)

    # Per-transform metrics
    avg_fragility_per_transform = {}
    avg_agreement_per_transform = {}
    avg_attn_sim_per_transform = {}

    for tname, data in all_fragilities_by_transform.items():
        avg_fragility_per_transform[tname] = float(np.mean(data["fragilities"])) if data["fragilities"] else 0
        avg_agreement_per_transform[tname] = float(np.mean(data["agreements"])) if data["agreements"] else 0
        avg_attn_sim_per_transform[tname] = float(np.mean(data["attention_sims"])) if data["attention_sims"] else None

    # Confusion matrix
    cm = None
    cm_labels = None
    if per_image_results:
        gt_labels = [r["ground_truth"] for r in per_image_results]
        pred_labels = [r["predicted_class"] for r in per_image_results]
        all_labels = sorted(set(gt_labels + pred_labels))
        try:
            cm = sk_confusion_matrix(gt_labels, pred_labels, labels=all_labels)
            cm = cm.tolist()
            cm_labels = all_labels
        except Exception:
            pass

    return {
        "accuracy": accuracy,
        "auroc": auroc,
        "high_trust_accuracy": high_trust_acc,
        "low_trust_accuracy": low_trust_acc,
        "total_images": total,
        "num_correct": num_correct,
        "avg_fragility_per_transform": avg_fragility_per_transform,
        "avg_agreement_per_transform": avg_agreement_per_transform,
        "avg_attn_sim_per_transform": avg_attn_sim_per_transform,
        "confusion_matrix": cm,
        "confusion_labels": cm_labels,
        "per_image_results": per_image_results,
        "note": "EXPLORATORY: Results based on small sample size. Not statistically definitive.",
    }


def _save_results(per_image_results, metrics, fragilities, output_dir):
    """Save evaluation results to CSV and JSON files."""
    # Per-image results
    if per_image_results:
        df = pd.DataFrame(per_image_results)
        df.to_csv(os.path.join(output_dir, "image_results.csv"), index=False)

    # Fragility results
    frag_rows = []
    for tname, data in fragilities.items():
        frag_rows.append({
            "transformation": tname,
            "avg_fragility": float(np.mean(data["fragilities"])) if data["fragilities"] else None,
            "avg_agreement": float(np.mean(data["agreements"])) if data["agreements"] else None,
            "avg_attention_similarity": float(np.mean(data["attention_sims"])) if data["attention_sims"] else None,
            "num_samples": len(data["fragilities"]),
        })
    if frag_rows:
        pd.DataFrame(frag_rows).to_csv(os.path.join(output_dir, "fragility_results.csv"), index=False)

    # Overall metrics
    metrics_to_save = {k: v for k, v in metrics.items()
                       if k not in ("per_image_results", "confusion_matrix", "confusion_labels")}
    with open(os.path.join(output_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics_to_save.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    writer.writerow([f"{k}.{sk}", sv])
            else:
                writer.writerow([k, v])

    # Experiment config
    config = {
        "timestamp": datetime.now().isoformat(),
        "num_images": len(per_image_results),
        "accuracy": metrics.get("accuracy"),
        "auroc": metrics.get("auroc"),
        "high_trust_accuracy": metrics.get("high_trust_accuracy"),
        "low_trust_accuracy": metrics.get("low_trust_accuracy"),
        "note": "EXPLORATORY RESULTS - small sample size",
    }
    with open(os.path.join(output_dir, "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"Results saved to {output_dir}")


def generate_confusion_matrix_figure(y_true, y_pred, class_names):
    """Generate a Plotly confusion matrix figure.

    Args:
        y_true: List of true labels.
        y_pred: List of predicted labels.
        class_names: List of class names.

    Returns:
        Plotly figure or None.
    """
    try:
        from sklearn.metrics import confusion_matrix as sk_confusion_matrix
        import plotly.express as px

        cm = sk_confusion_matrix(y_true, y_pred, labels=class_names)
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            x=class_names, y=class_names,
            labels=dict(x="Predicted", y="Actual", color="Count"),
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=500,
        )
        return fig
    except Exception as e:
        logger.error(f"Error generating confusion matrix: {e}")
        return None
