from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CATEGORY_COLS = ["blue_cast", "green_cast", "low_light", "blur"]
ATTN_COLS = [
    "attention_blue",
    "attention_green",
    "attention_lowlight",
    "attention_blur",
]


def normalize_name(name: str) -> str:
    return Path(str(name)).stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention-csv", default="results/attention_statistics.csv")
    parser.add_argument("--classification-csv", default="results/classification_result.csv")
    parser.add_argument("--output-csv", default="results/attention_by_category_f_100_r1.0_z_l.csv")
    args = parser.parse_args()

    attn = pd.read_csv(args.attention_csv)
    cls = pd.read_csv(args.classification_csv)

    attn["name_key"] = attn["image_name"].apply(normalize_name)

    if "image_name" in cls.columns:
        cls["name_key"] = cls["image_name"].apply(normalize_name)
    elif "filename" in cls.columns:
        cls["name_key"] = cls["filename"].apply(normalize_name)
    else:
        raise ValueError("classification_result.csv must contain image_name or filename column.")

    merged = attn.merge(cls, on="name_key", how="inner", suffixes=("_attn", "_cls"))

    rows = []

    for category in CATEGORY_COLS:
        if category not in merged.columns:
            print(f"Warning: {category} not found in classification csv, skipped.")
            continue

        subset = merged[merged[category].astype(bool)]

        if len(subset) == 0:
            continue

        row = {
            "category": category,
            "num_images": len(subset),
        }

        for col in ATTN_COLS:
            row[col] = subset[col].mean()

        dominant = max(ATTN_COLS, key=lambda c: row[c])
        row["dominant_attention"] = dominant.replace("attention_", "")

        rows.append(row)

    out = pd.DataFrame(rows)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(out)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()