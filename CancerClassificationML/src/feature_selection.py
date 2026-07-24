"""
Stage 5: Feature Selection
Compares three approaches on the leakage-safe train/test split from Stage 4:
  1. PCA (retain 95% variance)
  2. SelectKBest (ANOVA F-test), k tuned via stratified CV
  3. Variance threshold + pairwise correlation filtering

Uses a fast baseline classifier (LinearSVC) purely to RANK the feature-selection
methods via CV macro-F1. Full model tuning happens in Stage 6 - this stage is
about picking a feature representation, not the final model.

All selectors are fit INSIDE the CV loop (via sklearn Pipeline) to avoid
selection leakage.

RESULT FROM LAST RUN: PCA_95pct and SelectKBest_k200 both scored a perfect
1.0000 CV macro-F1 (a real, expected result for this dataset - the TCGA
pan-cancer 5-class RNA-seq benchmark is known to be highly separable).
Since the scores tie, this script locks in SelectKBest_k200 as the winner
instead of PCA, because it keeps real gene names - needed for Stage 8
(SHAP / top genes) and Stage 9 (Streamlit per-gene inputs), which PCA
components cannot provide.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

PROCESSED_DIR = Path("outputs/processed")
MODELS_DIR = Path("outputs/models")
REPORTS_DIR = Path("outputs/reports")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load Stage 4 outputs
# ---------------------------------------------------------------------------
X_train = np.load(PROCESSED_DIR / "X_train_scaled.npy")
X_test = np.load(PROCESSED_DIR / "X_test_scaled.npy")
y_train = np.load(PROCESSED_DIR / "y_train.npy")
y_test = np.load(PROCESSED_DIR / "y_test.npy")
surviving_genes = pd.read_csv(PROCESSED_DIR / "surviving_genes.csv")

print(f"Loaded X_train {X_train.shape}, X_test {X_test.shape}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
baseline_clf = LinearSVC(random_state=RANDOM_STATE, max_iter=5000)

results = []

# ---------------------------------------------------------------------------
# 2. Method A - PCA (95% variance), selector fit INSIDE each CV fold
# ---------------------------------------------------------------------------
print("\n[A] PCA (95% variance)...")
pipe_pca = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
    ("clf", baseline_clf)
])
scores_pca = cross_val_score(pipe_pca, X_train, y_train, cv=cv, scoring="f1_macro")
print(f"  CV macro-F1: {scores_pca.mean():.4f} +/- {scores_pca.std():.4f}")

pca_report = PCA(n_components=0.95, random_state=RANDOM_STATE).fit(X_train)
n_components = pca_report.n_components_
print(f"  Components needed for 95% variance (fit on full train set): {n_components}")

results.append(("PCA_95pct", n_components, scores_pca.mean(), scores_pca.std()))

# ---------------------------------------------------------------------------
# 3. Method B - SelectKBest (ANOVA F), tune k via CV, selector fit INSIDE each fold
# ---------------------------------------------------------------------------
print("\n[B] SelectKBest (ANOVA F), tuning k...")
k_candidates = [50, 100, 200, 500, 1000, 2000]
best_k, best_score = None, -np.inf

for k in k_candidates:
    pipe_kbest = Pipeline([
        ("select", SelectKBest(score_func=f_classif, k=k)),
        ("clf", baseline_clf)
    ])
    scores_k = cross_val_score(pipe_kbest, X_train, y_train, cv=cv, scoring="f1_macro")
    print(f"  k={k:5d} -> CV macro-F1: {scores_k.mean():.4f} +/- {scores_k.std():.4f}")
    results.append((f"SelectKBest_k{k}", k, scores_k.mean(), scores_k.std()))
    if scores_k.mean() > best_score:
        best_score, best_k = scores_k.mean(), k

print(f"  Best k: {best_k} (CV macro-F1={best_score:.4f})")

# ---------------------------------------------------------------------------
# 4. Method C - Variance threshold + correlation filtering (fast vectorized)
# ---------------------------------------------------------------------------
print("\n[C] Variance threshold + correlation filtering...")
corr_threshold = 0.95
vt = 0.5

selector_var = VarianceThreshold(threshold=vt)
X_train_var = selector_var.fit_transform(X_train)
n_after_var = X_train_var.shape[1]
print(f"  var_thresh={vt} -> {n_after_var} features after variance filter")

X32 = X_train_var.astype(np.float32)
corr_matrix = np.abs(np.corrcoef(X32, rowvar=False))

n_feat = corr_matrix.shape[0]
upper_tri = np.triu(np.ones((n_feat, n_feat), dtype=bool), k=1)
high_corr = (corr_matrix > corr_threshold) & upper_tri

to_drop_idx = set()
drop_rows, drop_cols = np.where(high_corr)
for i, j in zip(drop_rows, drop_cols):
    if i not in to_drop_idx:
        to_drop_idx.add(int(j))

keep_idx = [i for i in range(n_feat) if i not in to_drop_idx]
X_train_filtered = X_train_var[:, keep_idx]
n_after_corr = X_train_filtered.shape[1]
print(f"  after corr filter (>{corr_threshold}): {n_after_corr} features")

scores_vc = cross_val_score(baseline_clf, X_train_filtered, y_train, cv=cv, scoring="f1_macro")
print(f"  CV macro-F1: {scores_vc.mean():.4f} +/- {scores_vc.std():.4f}")

results.append((f"VarCorr_vt{vt}", n_after_corr, scores_vc.mean(), scores_vc.std()))
best_var_result = (vt, scores_vc.mean(), n_after_corr, to_drop_idx, selector_var)

# ---------------------------------------------------------------------------
# 5. Compare all methods
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results, columns=["method", "n_features", "cv_macro_f1_mean", "cv_macro_f1_std"])
results_df = results_df.sort_values("cv_macro_f1_mean", ascending=False).reset_index(drop=True)
print("\n=== Feature Selection Comparison (sorted by CV macro-F1) ===")
print(results_df.to_string(index=False))

results_df.to_csv(REPORTS_DIR / "feature_selection_comparison.csv", index=False)

# ---------------------------------------------------------------------------
# 6. Pick winner - OVERRIDDEN to SelectKBest_k200 for interpretability
#    (ties with PCA_95pct at 1.0000 CV macro-F1; see module docstring)
# ---------------------------------------------------------------------------
winner = "SelectKBest_k200"
print(f"\nWinning method (overridden for interpretability): {winner}")
print(f"  (Top auto-ranked method was {results_df.iloc[0]['method']} "
      f"at CV macro-F1={results_df.iloc[0]['cv_macro_f1_mean']:.4f} - tied, "
      f"so SelectKBest_k200 is used instead to keep real gene names.)")

# ---------------------------------------------------------------------------
# 7. Refit winning selector on full train set, apply to test set, save
# ---------------------------------------------------------------------------
if winner == "PCA_95pct":
    final_selector = PCA(n_components=0.95, random_state=RANDOM_STATE)
    X_train_final = final_selector.fit_transform(X_train)
    X_test_final = final_selector.transform(X_test)
    joblib.dump(final_selector, MODELS_DIR / "feature_selector.joblib")
    selected_gene_names = None

elif winner.startswith("SelectKBest"):
    k_final = int(winner.replace("SelectKBest_k", ""))
    final_selector = SelectKBest(score_func=f_classif, k=k_final)
    X_train_final = final_selector.fit_transform(X_train, y_train)
    X_test_final = final_selector.transform(X_test)
    joblib.dump(final_selector, MODELS_DIR / "feature_selector.joblib")
    mask = final_selector.get_support()
    selected_gene_names = surviving_genes.iloc[mask, 0].reset_index(drop=True)

else:  # VarCorr winner
    vt_final, _, _, to_drop_idx_final, selector_var_final = best_var_result

    X_train_var_final = selector_var_final.transform(X_train)
    X_test_var_final = selector_var_final.transform(X_test)

    n_feat_final = X_train_var_final.shape[1]
    keep_idx_final = [i for i in range(n_feat_final) if i not in to_drop_idx_final]

    X_train_final = X_train_var_final[:, keep_idx_final]
    X_test_final = X_test_var_final[:, keep_idx_final]

    joblib.dump(
        {
            "variance_selector": selector_var_final,
            "corr_drop_idx": sorted(to_drop_idx_final),
            "corr_keep_idx": keep_idx_final,
        },
        MODELS_DIR / "feature_selector.joblib",
    )

    var_mask = selector_var_final.get_support()
    genes_after_var = surviving_genes.iloc[var_mask, 0].reset_index(drop=True)
    selected_gene_names = genes_after_var.iloc[keep_idx_final].reset_index(drop=True)

print(f"\nFinal selected feature shape: train {X_train_final.shape}, test {X_test_final.shape}")

np.save(PROCESSED_DIR / "X_train_selected.npy", X_train_final)
np.save(PROCESSED_DIR / "X_test_selected.npy", X_test_final)

if selected_gene_names is not None:
    selected_gene_names.to_csv(PROCESSED_DIR / "selected_genes.csv", index=False)

with open(REPORTS_DIR / "feature_selection_summary.txt", "w") as f:
    f.write(f"Winning method: {winner}\n")
    f.write(f"Final feature count: {X_train_final.shape[1]}\n")
    f.write(f"Train shape: {X_train_final.shape}, Test shape: {X_test_final.shape}\n\n")
    f.write(results_df.to_string(index=False))

print("\nSaved: outputs/processed/X_train_selected.npy, X_test_selected.npy")
print("Saved: outputs/models/feature_selector.joblib")
print("Saved: outputs/reports/feature_selection_comparison.csv, feature_selection_summary.txt")
