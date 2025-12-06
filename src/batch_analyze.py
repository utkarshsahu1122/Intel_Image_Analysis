import json
import os
from typing import Optional, Dict, Any, Set

from tqdm import tqdm

from .vision_client import VisionService
from .dataset_loader import load_intel_samples


def load_processed_paths(output_path: str) -> Set[str]:
    """
    Read existing JSONL and collect image paths already processed.
    """
    processed: Set[str] = set()
    if not os.path.exists(output_path):
        return processed

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                img_path = obj.get("image_path")
                if img_path:
                    processed.add(img_path)
            except json.JSONDecodeError:
                continue

    return processed


def run_batch(
    max_images: Optional[int] = None,
    output_path: str = "data/processed/vision_results.jsonl",
) -> None:
    """
    Run Azure Vision on Intel dataset with resume capability.

    - If output_path already has results, previously processed images are skipped.
    - Results are appended to the same JSONL file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_records = load_intel_samples(max_images=max_images)
    print(f"✔ Candidate images loaded from dataset: {len(all_records)}")

    already_done = load_processed_paths(output_path)
    print(f"✔ Already processed (from JSONL): {len(already_done)}")

    records = [r for r in all_records if r["image_path"] not in already_done]
    print(f"✔ Remaining to process: {len(records)}")

    if not records:
        print("✔ Nothing new to process. Exiting.")
        return

    vision = VisionService()

    # append to existing file
    with open(output_path, "a", encoding="utf-8") as f_out:
        for rec in tqdm(records, desc="Analyzing images with Azure Vision"):
            path = rec["image_path"]
            label = rec["label"]

            try:
                analysis: Dict[str, Any] = vision.analyze_file(path)
            except Exception as e:
                print(f"[WARN] Failed for {path}: {e}")
                continue

            row = {
                "image_path": path,
                "label": label,
                "tags": analysis.get("tags", []),
                "caption": analysis.get("caption", None),
                "metadata": analysis.get("metadata", {}),
            }
            f_out.write(json.dumps(row) + "\n")
            f_out.flush()


if __name__ == "__main__":
    # For initial subset:
    # run_batch(max_images=1000)

    # For full run (all available images from seg_train):
    run_batch(max_images=None)
