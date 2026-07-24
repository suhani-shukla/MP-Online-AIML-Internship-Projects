"""
Stage 9a: Retrain a lightweight model on the top 25 consolidated genes,
compare accuracy/macro-F1 against the full 200-gene model, and save
a self-contained deployment bundle (model + scaler subset + labels)
for the Streamlit app.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---- Load existing artifacts ----
X_train_200 = np.load("outputs/processed/X_train_selected.npy")
X_test_200 = np.load("outputs/processed/X_test_selected.npy")
y_train = np.load("outputs/processed/y_train.npy")
y_test = np.load("outputs/processed/y_test.npy")

selected_genes = pd.read_csv("outputs/processed/selected_genes.csv")["gene"].tolist()
consolidated = pd.read_csv("outputs/reports/top_genes_consolidated.csv")

TOP_N = 25
top_genes = consolidated.head(TOP_N)["gene"].tolist()

print(f"Loaded 200-gene data: X_train {X_train_200.shape}, X_test {X_test_200.shape}")
print(f"Selecting top {len(top_genes)} genes from consolidated interpretability ranking")

# Map top genes -> column indices within the 200-gene selected set
gene_to_idx = {g: i for i, g in enumerate(selected_genes)}
missing = [g for g in top_genes if g not in gene_to_idx]
if missing:
    raise ValueError(f"Genes not found in selected_genes.csv: {missing}")

top_idx = [gene_to_idx[g] for g in top_genes]

X_train_25 = X_train_200[:, top_idx]
X_test_25 = X_test_200[:, top_idx]

print(f"Subset shape: X_train {X_train_25.shape}, X_test {X_test_25.shape}")

# ---- Retrain MLP (same family as Stage 6 best model) on 25 genes ----
best_model_200 = joblib.load("outputs/models/best_model.joblib")
try:
    params = best_model_200.get_params()
    mlp_params = {k: v for k, v in params.items() if k in
                  ["hidden_layer_sizes", "activation", "alpha", "learning_rate_init",
                   "max_iter", "early_stopping", "random_state"]}
except Exception:
    mlp_params = {}

mlp_params.setdefault("max_iter", 2000)
mlp_params.setdefault("random_state", 42)

print(f"Retraining MLPClassifier on 25 genes with params: {mlp_params}")
model_25 = MLPClassifier(**mlp_params)
model_25.fit(X_train_25, y_train)

# ---- Evaluate: 25-gene model vs 200-gene model on same test set ----
y_pred_25 = model_25.predict(X_test_25)
y_pred_200 = best_model_200.predict(X_test_200)

acc_25 = accuracy_score(y_test, y_pred_25)
f1_25 = f1_score(y_test, y_pred_25, average="macro")
acc_200 = accuracy_score(y_test, y_pred_200)
f1_200 = f1_score(y_test, y_pred_200, average="macro")

print("\n=== Accuracy / Macro-F1 tradeoff ===")
print(f"200-gene model : acc={acc_200:.4f}  macro-F1={f1_200:.4f}")
print(f"25-gene  model : acc={acc_25:.4f}  macro-F1={f1_25:.4f}")

label_encoder = joblib.load("outputs/models/label_encoder.joblib")
print("\nClassification report (25-gene model):")
print(classification_report(y_test, y_pred_25, target_names=label_encoder.classes_))

# ---- Save tradeoff summary ----
tradeoff = pd.DataFrame([
    {"model": "200-gene (full)", "n_genes": 200, "accuracy": acc_200, "macro_f1": f1_200},
    {"model": "25-gene (lightweight)", "n_genes": 25, "accuracy": acc_25, "macro_f1": f1_25},
])
tradeoff.to_csv("outputs/reports/lightweight_vs_full_tradeoff.csv", index=False)
print("\nSaved: outputs/reports/lightweight_vs_full_tradeoff.csv")

# ---- Build deployment-ready preprocessing for the 25 raw genes ----
# App will take RAW expression values -> log2(x+1) -> scale using just
# these 25 genes' mean_/scale_ from the Stage 4 StandardScaler -> predict.
scaler_full = joblib.load("outputs/models/scaler.joblib")
surviving_genes = pd.read_csv("outputs/processed/surviving_genes.csv")["gene"].tolist()
gene_pos_in_scaler = {g: i for i, g in enumerate(surviving_genes)}

missing_scaler = [g for g in top_genes if g not in gene_pos_in_scaler]
if missing_scaler:
    raise ValueError(f"Genes not found in surviving_genes.csv: {missing_scaler}")

scaler_idx = [gene_pos_in_scaler[g] for g in top_genes]
mean_25 = scaler_full.mean_[scaler_idx]
scale_25 = scaler_full.scale_[scaler_idx]

deployment_bundle = {
    "model": model_25,
    "gene_order": top_genes,
    "mean_": mean_25,
    "scale_": scale_25,
    "label_classes": list(label_encoder.classes_),
}

joblib.dump(deployment_bundle, "outputs/models/deployment_bundle.joblib")
print("Saved: outputs/models/deployment_bundle.joblib (model + 25-gene scaler subset + labels)")
