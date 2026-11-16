"""
Test dataset loading and evaluation pipeline on 5 samples
"""
import sys
import os
sys.path.insert(0, ".")

from model import load_model, predict
from transformations import ALL_TRANSFORMS, MILD_TRANSFORMS
from attention import attention_rollout, compute_attention_similarity
from dataset import load_imagenette
from evaluation import evaluate_dataset

def test_eval():
    print("Testing Imagenette load (20 samples)...")
    samples = load_imagenette(num_samples=20, seed=42)
    print(f"Loaded {len(samples)} samples.")

    print("\nRunning evaluation on 20 samples...")
    output_dir = os.path.join("outputs")
    metrics = evaluate_dataset(
        images_with_labels=samples,
        predict_fn=predict,
        attention_rollout_fn=attention_rollout,
        attention_similarity_fn=compute_attention_similarity,
        transforms_dict=ALL_TRANSFORMS,
        mild_transform_names=list(MILD_TRANSFORMS.keys()),
        output_dir=output_dir,
    )
    print("\nEvaluation metrics:")
    print(f"  Total Images: {metrics['total_images']}")
    print(f"  Accuracy:     {metrics['accuracy']*100:.1f}%")
    print(f"  AUROC:        {metrics['auroc']}")
    print(f"  High-Trust Acc: {metrics['high_trust_accuracy']}")
    print(f"  Low-Trust Acc:  {metrics['low_trust_accuracy']}")
    print("Files created in outputs/:")
    for f in os.listdir(output_dir):
        print("  -", f)

if __name__ == "__main__":
    test_eval()
