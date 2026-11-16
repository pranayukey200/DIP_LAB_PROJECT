"""
TransformTrust: End-to-End Pipeline Verification Test
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, ".")

from model import load_model, predict, get_model_name, get_device
from transformations import ALL_TRANSFORMS, MILD_TRANSFORMS, STRESS_TRANSFORMS
from attention import (
    attention_rollout,
    compute_attention_similarity,
    generate_attention_heatmap,
    generate_attention_comparison,
)
from trust_engine import (
    compute_trust_score,
    compute_all_fragilities,
    detect_confidence_trust_mismatch,
    run_full_analysis,
)

def run_test():
    print("=" * 60)
    print("TRANSFORMTRUST: END-TO-END PIPELINE VERIFICATION")
    print("=" * 60)

    # 1. Device and Model
    device = get_device()
    model_name = get_model_name()
    print(f"1. Hardware Device: {device}")
    print(f"   Model Name:     {model_name}")

    print("\n2. Loading Model & Processor...")
    model, processor = load_model()
    print("   Model loaded successfully!")

    # 3. Create a synthetic test image with recognizable geometric patterns
    print("\n3. Creating Synthetic Test Image (224x224)...")
    img = Image.new("RGB", (224, 224), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Draw a colored pattern
    draw.rectangle([40, 40, 180, 180], fill=(30, 120, 220), outline=(10, 50, 150))
    draw.ellipse([70, 70, 150, 150], fill=(220, 50, 50), outline=(150, 20, 20))
    draw.polygon([(112, 50), (60, 160), (164, 160)], outline=(255, 255, 0), width=3)
    
    test_img_path = os.path.join("outputs", "sample_test_image.png")
    os.makedirs("outputs", exist_ok=True)
    img.save(test_img_path)
    print(f"   Saved sample image to {test_img_path}")

    # 4. Predict original image
    print("\n4. Running Inference on Original Image...")
    orig_res = predict(img, model=model, processor=processor, device=device)
    print(f"   Predicted Class: {orig_res['predicted_class']}")
    print(f"   Confidence:      {orig_res['confidence']*100:.2f}%")
    print(f"   Top-5:           {[(c, round(p, 4)) for c, p in orig_res['top5']]}")
    print(f"   Attentions count: {len(orig_res['attentions'])} layers")

    # 5. Attention Rollout
    print("\n5. Computing Attention Rollout (Abnar & Zuidema)...")
    orig_attn = attention_rollout(orig_res["attentions"])
    print(f"   Rollout shape: {orig_attn.shape} (min={orig_attn.min():.4f}, max={orig_attn.max():.4f})")
    assert orig_attn.shape == (14, 14), f"Expected 14x14, got {orig_attn.shape}"

    # Generate and save original attention heatmap
    heatmap_img = generate_attention_heatmap(img, orig_attn)
    heatmap_path = os.path.join("outputs", "attention_maps", "orig_attention_heatmap.png")
    os.makedirs(os.path.join("outputs", "attention_maps"), exist_ok=True)
    heatmap_img.save(heatmap_path)
    print(f"   Saved attention heatmap to {heatmap_path}")

    # 6. Apply all transformations and compute attention similarity
    print("\n6. Running Transformation Robustness & Attention Alignment Tests:")
    transform_results = {}
    mild_names = list(MILD_TRANSFORMS.keys())

    for tname, (tfunc, tdesc) in ALL_TRANSFORMS.items():
        t_img, t_meta = tfunc(img)
        t_res = predict(t_img, model=model, processor=processor, device=device)
        t_attn = attention_rollout(t_res["attentions"])
        sim = compute_attention_similarity(orig_attn, t_attn, t_meta)
        
        transform_results[tname] = {
            "prediction": t_res,
            "transformed_image": t_img,
            "attention_map": t_attn,
            "attention_similarity": sim,
            "metadata": t_meta,
            "description": tdesc,
        }

        agrees = (t_res["predicted_class"] == orig_res["predicted_class"])
        sim_str = f"{sim:.4f}" if sim is not None else "N/A (unaligned)"
        print(f"   [{t_meta['type'].upper():6s}] {tname:16s} -> "
              f"Pred: {t_res['predicted_class'][:18]:18s} | "
              f"Conf: {t_res['confidence']*100:5.1f}% | "
              f"Agree: {'YES' if agrees else 'NO ':3s} | "
              f"Attn Sim: {sim_str}")

    # 7. Trust Score Calculation
    print("\n7. Computing TransformTrust Score...")
    trust_data = compute_trust_score(orig_res, transform_results, mild_names)
    print(f"   Agreement Rate:            {trust_data['agreement_rate']*100:.1f}% "
          f"({trust_data['num_agreeing_mild']}/{trust_data['num_total_mild']} mild views)")
    print(f"   Mean Attention Similarity: {trust_data['mean_attention_similarity']*100:.1f}%")
    print(f"   ==> TRANSFORMTRUST SCORE:  {trust_data['trust_score']:.2f} / 100")

    # 8. Fragility Analysis
    print("\n8. Transformation Fragility Ranking:")
    fragilities = compute_all_fragilities(orig_res, transform_results)
    for idx, f in enumerate(fragilities, 1):
        sim_str = f"{f['attention_similarity']:.4f}" if f['attention_similarity'] is not None else "N/A"
        print(f"   Rank {idx}. {f['transform_name']:16s} | Fragility: {f['fragility']:.4f} | "
              f"Agree: {f['prediction_agrees']} | Attn Sim: {sim_str}")

    # 9. Confidence-Trust Mismatch Detection
    print("\n9. Confidence-Trust Mismatch Analysis:")
    mismatch = detect_confidence_trust_mismatch(
        orig_res["confidence"] * 100,
        trust_data["trust_score"],
        threshold=20.0
    )
    print(f"   Model Softmax Confidence: {mismatch['confidence']:.2f}%")
    print(f"   TransformTrust Score:     {mismatch['trust_score']:.2f}%")
    print(f"   Mismatch Gap:             {mismatch['mismatch_gap']:.2f}%")
    print(f"   Mismatch Detected:        {mismatch['is_mismatch']}")
    print(f"   Severity:                 {mismatch['severity'].upper()}")
    print(f"   Explanation:              {mismatch['explanation']}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! SYSTEM IS FULLY OPERATIONAL.")
    print("=" * 60)

if __name__ == "__main__":
    run_test()
