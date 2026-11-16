# 🛡️ TransformTrust

## Explainable and Confidence-Aware Reliability Analysis of Vision Transformers

TransformTrust is an intelligent image classification and reliability analysis application that evaluates how reliable Vision Transformer (ViT) predictions are when input images undergo different transformations.

### Core Research Question

> *"How reliable are Vision Transformer predictions when input images undergo different transformations, and can prediction consistency and attention stability provide useful signals of reliability?"*

### Key Features

- **Transformation-Based Prediction Consistency**: Tests model robustness across 8 image transformations
- **Attention Rollout Analysis**: Implements Abnar & Zuidema attention rollout for visualization
- **TransformTrust Score**: Custom reliability score combining prediction consistency and attention stability
- **Fragility Analysis**: Identifies which transformations most disturb the model
- **Confidence-Trust Mismatch Detection**: Flags cases where model confidence doesn't match reliability

### Technical Stack

- **Model**: google/vit-base-patch16-224 (GPU) / facebook/deit-tiny-patch16-224 (CPU)
- **Framework**: PyTorch + Hugging Face Transformers
- **UI**: Streamlit + Plotly
- **Dataset**: Imagenette (10-class subset of ImageNet)

### Project Structure

```
TransformTrust/
├── app.py                 # Streamlit UI application
├── model.py               # ViT model loading and inference
├── transformations.py     # Image transformation engine
├── attention.py           # Attention rollout implementation
├── trust_engine.py        # Trust score and fragility computation
├── evaluation.py          # Dataset evaluation pipeline
├── dataset.py             # Imagenette dataset loader
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── outputs/               # Evaluation results
│   ├── metrics.csv
│   ├── image_results.csv
│   ├── fragility_results.csv
│   ├── attention_maps/
│   └── charts/
└── saved_model/           # Cached model files
```

### Installation

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

### Trust Score Formula

```
Trust Score = 0.5 × Agreement Rate + 0.5 × Mean Attention Similarity
```

Where:
- **Agreement Rate** = Fraction of mild transformations retaining the original top-1 prediction
- **Mean Attention Similarity** = Average cosine similarity between original and aligned transformed attention maps

### Transformations

**Mild (Label-Preserving):**
- Gaussian Blur
- Grayscale
- Brightness Modification
- Contrast Modification
- Mild Occlusion

**Structural Stress Tests:**
- 8-pixel Sub-patch Shift
- Patch Shuffle
- Patch Dropout (25%)

### Important Notes

- The Trust Score is a **heuristic consistency index**, not a calibrated probability of correctness
- Attention rollout is a **heuristic visualization**, not a definitive explanation of model reasoning
- Confidence-Trust mismatch signals warrant further inspection but do not prove predictions are incorrect
- Evaluation results on small samples are **exploratory**

### References

- Abnar, S., & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. ACL.
- Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words. ICLR.
- Howard, J. (2019). Imagenette: A smaller subset of 10 easily classified classes from ImageNet.
