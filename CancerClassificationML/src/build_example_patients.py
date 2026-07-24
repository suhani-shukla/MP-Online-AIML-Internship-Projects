"""
Stage 9b: Build UI support data for app.py —
- Real example patients with actual raw expression values
- Slider min/max/default per gene, using true min/max (+ small padding)
  so any real patient value always falls within slider range
"""

import pandas as pd
import numpy as np
import json

TOP_N = 25
SEED = 42

consolidated = pd.read_csv("outputs/reports/top_genes_consolidated.csv")
top_genes = consolidated.head(TOP_N)["gene"].tolist()

print("Loading raw_data...")
data = pd.read_csv("raw_data/data.csv", index_col=0)
labels = pd.read_csv("raw_data/labels.csv", index_col=0)
merged = data.join(labels)

subset = merged[top_genes + ["Class"]]
print(f"Subset shape: {subset.shape}")

# ---- Slider ranges: true min/max + 5% padding, so no real value can exceed bounds ----
slider_info = {}
for gene in top_genes:
    vals = subset[gene].values
    true_min, true_max = float(vals.min()), float(vals.max())
    span = true_max - true_min
    pad = span * 0.05 if span > 0 else 1.0
    slider_info[gene] = {
        "min": max(0.0, true_min - pad),
        "max": true_max + pad,
        "default": float(np.median(vals)),
    }

# ---- Pick 2 real example patients per class ----
examples = {}
for cls, group in subset.groupby("Class"):
    picks = group.sample(n=min(2, len(group)), random_state=SEED)
    examples[cls] = [
        {"sample_id": str(idx), "values": {g: float(row[g]) for g in top_genes}}
        for idx, row in picks.iterrows()
    ]
    print(f"{cls}: picked {len(picks)} example patient(s)")

ui_data = {
    "gene_order": top_genes,
    "slider_info": slider_info,
    "examples": examples,
}

with open("outputs/models/app_ui_data.json", "w") as f:
    json.dump(ui_data, f, indent=2)

print("\nSaved: outputs/models/app_ui_data.json")
