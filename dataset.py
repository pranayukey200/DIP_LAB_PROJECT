"""
TransformTrust: Dataset Module
================================
Handles loading Imagenette (10-class subset of ImageNet) for evaluation.
Maps Imagenette folder names to correct ImageNet class indices.
"""

from __future__ import annotations

import os
import logging
import random
from typing import List, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# Imagenette classes mapped to ImageNet class indices and human-readable names
IMAGENETTE_CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}

# ImageNet label indices for each Imagenette class
IMAGENETTE_TO_IMAGENET = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}

# Reverse mapping: ImageNet index -> human readable name
IMAGENET_IDX_TO_NAME = {v: IMAGENETTE_CLASSES[k] for k, v in IMAGENETTE_TO_IMAGENET.items()}


def load_imagenette(
    num_samples: int = 50,
    seed: int = 42,
) -> List[Tuple[Image.Image, str, int]]:
    """Load a subset of the Imagenette validation set.

    Args:
        num_samples: Number of images to load.
        seed: Random seed for reproducible subsampling.

    Returns:
        List of (PIL.Image, ground_truth_label_str, imagenet_label_index) tuples.
    """
    cache_dir = os.path.join(os.path.dirname(__file__), "saved_model", "imagenette_cache")
    extract_dir = os.path.join(cache_dir, "imagenette2-160")
    if os.path.exists(os.path.join(extract_dir, "val")):
        return _load_from_local_dir(extract_dir, num_samples, seed)

    try:
        return _load_via_download(num_samples, seed)
    except Exception as e:
        logger.warning(f"Download fallback failed: {e}. Trying datasets library...")
        return _load_via_datasets(num_samples, seed)


def _load_from_local_dir(extract_dir: str, num_samples: int, seed: int) -> List[Tuple[Image.Image, str, int]]:
    """Load directly from local extracted imagenette2-160 directory."""
    val_dir = os.path.join(extract_dir, "val")
    results = []

    for wnid in IMAGENETTE_CLASSES:
        class_dir = os.path.join(val_dir, wnid)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(class_dir, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                    human_name = IMAGENETTE_CLASSES[wnid]
                    imagenet_idx = IMAGENETTE_TO_IMAGENET[wnid]
                    results.append((img, human_name, imagenet_idx))
                except Exception:
                    continue

    rng = random.Random(seed)
    rng.shuffle(results)
    results = results[:num_samples]
    logger.info(f"Loaded {len(results)} Imagenette images from local cache.")
    return results


def _load_via_datasets(num_samples: int, seed: int) -> List[Tuple[Image.Image, str, int]]:
    """Load Imagenette using the HuggingFace datasets library."""
    from datasets import load_dataset

    logger.info("Loading Imagenette via HuggingFace datasets...")
    ds = load_dataset("frgfm/imagenette", "160px", split="validation", trust_remote_code=True)

    # The dataset has 'image' (PIL) and 'label' (int index within imagenette classes)
    # Map imagenette label indices to our class info
    # frgfm/imagenette label order: 0-9 corresponding to the 10 classes
    imagenette_wnids = list(IMAGENETTE_CLASSES.keys())

    # Subsample
    total = len(ds)
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    selected = indices[:min(num_samples, total)]

    results = []
    for idx in selected:
        item = ds[idx]
        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        label_idx = item["label"]
        wnid = imagenette_wnids[label_idx]
        human_name = IMAGENETTE_CLASSES[wnid]
        imagenet_idx = IMAGENETTE_TO_IMAGENET[wnid]

        results.append((img, human_name, imagenet_idx))

    logger.info(f"Loaded {len(results)} Imagenette validation images.")
    return results


def _load_via_download(num_samples: int, seed: int) -> List[Tuple[Image.Image, str, int]]:
    """Fallback: download Imagenette directly."""
    import io
    import os
    import tarfile
    import urllib.request

    url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    cache_dir = os.path.join(os.path.dirname(__file__), "saved_model", "imagenette_cache")
    extract_dir = os.path.join(cache_dir, "imagenette2-160")

    if not os.path.exists(extract_dir):
        os.makedirs(cache_dir, exist_ok=True)
        tgz_path = os.path.join(cache_dir, "imagenette2-160.tgz")

        if not os.path.exists(tgz_path):
            logger.info("Downloading Imagenette2-160...")
            urllib.request.urlretrieve(url, tgz_path)

        logger.info("Extracting Imagenette2-160...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(cache_dir)

    val_dir = os.path.join(extract_dir, "val")
    results = []

    for wnid in IMAGENETTE_CLASSES:
        class_dir = os.path.join(val_dir, wnid)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(class_dir, fname)
                try:
                    img = Image.open(fpath).convert("RGB")
                    human_name = IMAGENETTE_CLASSES[wnid]
                    imagenet_idx = IMAGENETTE_TO_IMAGENET[wnid]
                    results.append((img, human_name, imagenet_idx))
                except Exception:
                    continue

    rng = random.Random(seed)
    rng.shuffle(results)
    results = results[:num_samples]
    logger.info(f"Loaded {len(results)} Imagenette images via direct download.")
    return results
