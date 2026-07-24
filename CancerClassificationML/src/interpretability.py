"""
Stage 8: Interpretability
Identifies the top 20-30 genes driving predictions, using three methods:
  1. Native feature_importances_ from RandomForest and XGBoost
  2. SHAP TreeExplainer values for RandomForest and XGBoost (fast, exact)
  3. Permutation importance on MLPClassifier (the actual best model), since
     MLP has no native feature_importances_ and SHAP's KernelExplainer would
     be slow/approximate for a neural net. Permutation importance works on
     any model and directly measures each gene's effect on the real deployed
     model's real predictions.

Consolidates all three into one top-genes table so genes that agree across
methods can be highlighted as the most trustworthy findings.
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

import shap
from sklearn.inspection import permutation_importance

PROCESSED_DIR = Path("outputs/processed")
MODELS_DIR = Path("outputs/models")
FIGURES_DIR = Path("outputs/figures")
REPORTS_DIR = Path("outputs/reports")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TOP_N = 25

# ---------------------------------------------------------------------------
# 1. Load data, models, gene names
# ---------------------------------------------------------------------------
X_train = np.load(PROCESSED_DIR / "X_train_selected.npy")
X_test = np.load(PROCESSED_DIR / "X_test_selected.npy")
y_train = np.load(PROCESSED_DIR / "y_train.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")

gene_names = pd.read_csv(PROCESSED_DIR / "selected_genes.csv").iloc[:, 0].tolist()
assert len(gene_names) == X_train.shape[1], "Gene name count doesn't match feature count!"

rf_model = joblib.load(MODELS_DIR / "model_RandomForest.joblib")
xgb_model = joblib.load(MODELS_DIR / "model_XGBoost.joblib")
mlp_model = joblib.load(MODELS_DIR / "model_MLPClassifier.joblib")

print(f"Loaded {len(gene_names)} gene names, X_train {X_train.shape}, X_test {X_test.shape}")

# ---------------------------------------------------------------------------
# 2. Native feature_importances_ - RandomForest
# ---------------------------------------------------------------------------
print("\n[1/3] RandomForest feature_importances_...")
rf_importances = rf_model.feature_importances_
rf_top = pd.DataFrame({
    "gene": gene_names,
    "rf_importance": rf_importances
}).sort_values("rf_importance", ascending=False).reset_index(drop=True)
print(rf_top.head(TOP_N).to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Native feature_importances_ - XGBoost
# ---------------------------------------------------------------------------
print("\n[2/3] XGBoost feature_importances_...")
xgb_importances = xgb_model.feature_importances_
xgb_top = pd.DataFrame({
    "gene": gene_names,
    "xgb_importance": xgb_importances
}).sort_values("xgb_importance", ascending=False).reset_index(drop=True)
print(xgb_top.head(TOP_N).to_string(index=False))

# ---------------------------------------------------------------------------
# 4. SHAP TreeExplainer - RandomForest and XGBoost
# ---------------------------------------------------------------------------
print("\n[3/3 - part A] SHAP values for RandomForest...")
rf_explainer = shap.TreeExplainer(rf_model)
rf_shap_values = rf_explainer.shap_values(X_test)
# shap_values is a list of arrays (one per class) for multiclass RF; average abs across classes
if isinstance(rf_shap_values, list):
    rf_shap_mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in rf_shap_values], axis=0)
else:
    rf_shap_mean_abs = np.abs(rf_shap_values).mean(axis=(0, 2))

rf_shap_df = pd.DataFrame({
    "gene": gene_names,
    "rf_shap_importance": rf_shap_mean_abs
}).sort_values("rf_shap_importance", ascending=False).reset_index(drop=True)

print("\nSHAP values for XGBoost...")
xgb_explainer = shap.TreeExplainer(xgb_model)
xgb_shap_values = xgb_explainer.shap_values(X_test)
if isinstance(xgb_shap_values, list):
    xgb_shap_mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in xgb_shap_values], axis=0)
else:
    xgb_shap_mean_abs = np.abs(xgb_shap_values).mean(axis=(0, 2)) if xgb_shap_values.ndim == 3 else np.abs(xgb_shap_values).mean(axis=0)

xgb_shap_df = pd.DataFrame({
    "gene": gene_names,
    "xgb_shap_importance": xgb_shap_mean_abs
}).sort_values("xgb_shap_importance", ascending=False).reset_index(drop=True)

# SHAP summary plot (RandomForest, as it's the more standard choice for this plot type)
plt.figure()
shap.summary_plot(rf_shap_values, X_test, feature_names=gene_names, show=False, max_display=TOP_N)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "shap_summary_randomforest.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved SHAP summary plot: outputs/figures/shap_summary_randomforest.png")

# ---------------------------------------------------------------------------
# 5. Permutation importance - MLPClassifier (the actual best/deployed model)
# ---------------------------------------------------------------------------
print("\n[3/3 - part B] Permutation importance for MLPClassifier (best model)...")
perm_result = permutation_importance(
    mlp_model, X_test, y_test,
    scoring="f1_macro", n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1
)
mlp_perm_df = pd.DataFrame({
    "gene": gene_names,
    "mlp_perm_importance_mean": perm_result.importances_mean,
    "mlp_perm_importance_std": perm_result.importances_std,
}).sort_values("mlp_perm_importance_mean", ascending=False).reset_index(drop=True)
print(mlp_perm_df.head(TOP_N).to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Consolidate: merge all rankings, count how many top-N lists each gene appears in
# ---------------------------------------------------------------------------
print("\nConsolidating rankings across all methods...")

top_genes_sets = {
    "RF_importance": set(rf_top.head(TOP_N)["gene"]),
    "XGB_importance": set(xgb_top.head(TOP_N)["gene"]),
    "RF_SHAP": set(rf_shap_df.head(TOP_N)["gene"]),
    "XGB_SHAP": set(xgb_shap_df.head(TOP_N)["gene"]),
    "MLP_permutation": set(mlp_perm_df.head(TOP_N)["gene"]),
}

all_genes = set(gene_names)
consolidated = pd.DataFrame({"gene": list(all_genes)})
for method_name, gene_set in top_genes_sets.items():
    consolidated[f"in_top{TOP_N}_{method_name}"] = consolidated["gene"].isin(gene_set)

consolidated["agreement_count"] = consolidated[
    [c for c in consolidated.columns if c.startswith("in_top")]
].sum(axis=1)

# Merge in the actual importance scores for reference
consolidated = consolidated.merge(rf_top, on="gene", how="left")
consolidated = consolidated.merge(xgb_top, on="gene", how="left")
consolidated = consolidated.merge(rf_shap_df, on="gene", how="left")
consolidated = consolidated.merge(xgb_shap_df, on="gene", how="left")
consolidated = consolidated.merge(mlp_perm_df[["gene", "mlp_perm_importance_mean"]], on="gene", how="left")

consolidated = consolidated.sort_values("agreement_count", ascending=False).reset_index(drop=True)

print(f"\n=== Top genes by cross-method agreement (out of 5 methods) ===")
print(consolidated.head(TOP_N)[["gene", "agreement_count", "rf_importance", "xgb_importance", "mlp_perm_importance_mean"]].to_string(index=False))

n_unanimous = (consolidated["agreement_count"] == 5).sum()
n_majority = (consolidated["agreement_count"] >= 3).sum()
print(f"\nGenes appearing in ALL 5 top-{TOP_N} lists: {n_unanimous}")
print(f"Genes appearing in >=3 of 5 top-{TOP_N} lists: {n_majority}")

# ---------------------------------------------------------------------------
# 7. Save everything
# ---------------------------------------------------------------------------
consolidated.to_csv(REPORTS_DIR / "top_genes_consolidated.csv", index=False)
rf_top.head(TOP_N).to_csv(REPORTS_DIR / "top_genes_rf_importance.csv", index=False)
xgb_top.head(TOP_N).to_csv(REPORTS_DIR / "top_genes_xgb_importance.csv", index=False)
mlp_perm_df.head(TOP_N).to_csv(REPORTS_DIR / "top_genes_mlp_permutation.csv", index=False)

# Bar chart of the top consolidated genes (by agreement, then by MLP permutation importance)
top_for_plot = consolidated.head(TOP_N).sort_values("mlp_perm_importance_mean", ascending=True)
plt.figure(figsize=(8, 10))
plt.barh(top_for_plot["gene"], top_for_plot["mlp_perm_importance_mean"], color="steelblue")
plt.xlabel("Permutation importance (MLP, drop in macro-F1)")
plt.title(f"Top {TOP_N} genes by cross-method agreement\n(bar length = MLP permutation importance)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "top_genes_consolidated_barplot.png", dpi=150)
plt.close()

print(f"\nSaved: outputs/reports/top_genes_consolidated.csv")
print(f"Saved: outputs/reports/top_genes_rf_importance.csv, top_genes_xgb_importance.csv, top_genes_mlp_permutation.csv")
print(f"Saved: outputs/figures/shap_summary_randomforest.png")
print(f"Saved: outputs/figures/top_genes_consolidated_barplot.png")
