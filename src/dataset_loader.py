import os
import random
from typing import List, Dict, Optional


def load_intel_samples(
    root_dir: str = "data/raw/intel/seg_train",
    max_images: Optional[int] = None,
) -> List[Dict]:
    """
    Load up to `max_images` images from the Intel scene classification dataset.

    Returns: list of dicts { "image_path": str, "label": str }
    """
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(
            f"{root_dir} not found. Make sure you unzipped the Intel dataset correctly."
        )

    class_dirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    records: List[Dict] = []

    for cls in class_dirs:
        cls_dir = os.path.join(root_dir, cls)
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            records.append(
                {
                    "image_path": os.path.join(cls_dir, fname),
                    "label": cls,
                }
            )

    random.shuffle(records)

    if max_images is not None and max_images < len(records):
        records = records[:max_images]

    return records
