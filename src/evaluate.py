import json
from collections import Counter

import pandas as pd

JSONL_PATH = "data/processed/vision_results.jsonl"


def load_results(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    print(f"Loading results from {JSONL_PATH} ...")
    df = load_results(JSONL_PATH)
    print(f"Loaded {len(df)} rows")

    # Expecting columns: image_path, label, tags, caption, metadata
    if "label" not in df.columns or "tags" not in df.columns:
        raise ValueError(
            "Expected columns 'label' and 'tags' in results. "
            f"Found columns: {list(df.columns)}"
        )

    # Extract only tag names
    df["tag_names"] = df["tags"].apply(
        lambda tags: [str(t.get("name", "")).lower() for t in tags]
        if isinstance(tags, list)
        else []
    )

    # Simple metric: does the class name appear in any tag string?
    def has_label_in_tags(row):
        label = str(row["label"]).lower()
        tags = row["tag_names"]
        # class name contained in tag OR tag name contained in class name
        return any(label in t or t in label for t in tags)

    df["class_match"] = df.apply(has_label_in_tags, axis=1)
    overall_rate = df["class_match"].mean()
    print(f"\nOverall tag-class match rate: {overall_rate * 100:.1f}%")

    # Per-class match rate
    print("\nPer-class match rate:")
    class_stats = (
        df.groupby("label")["class_match"]
        .mean()
        .sort_values(ascending=False)
    )
    for label, rate in class_stats.items():
        print(f"  {label:10s}: {rate * 100:5.1f}%")

    # Top 20 tags overall
    tag_counter = Counter()
    for tags in df["tag_names"]:
        tag_counter.update(tags)

    print("\nTop 20 tags across all images:")
    for name, cnt in tag_counter.most_common(20):
        print(f"  {name:20s}: {cnt}")

    # Show a few sample rows for manual inspection
    print("\nSample rows (image_path, label, first 5 tags, caption):")
    for _, row in df.head(5).iterrows():
        print("-" * 60)
        print("Image:", row["image_path"])
        print("Label:", row["label"])
        print("Tags :", row["tag_names"][:5])
        print("Caption:", (row.get("caption") or {}).get("text") if isinstance(row.get("caption"), dict) else row.get("caption"))


if __name__ == "__main__":
    main()
