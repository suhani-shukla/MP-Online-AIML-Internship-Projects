"""
Stage 3: Load, merge, and explore the Gene Expression Cancer RNA-Seq dataset.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = "raw_data/data.csv"
LABELS_PATH = "raw_data/labels.csv"
FIG_DIR = "outputs/figures"
REPORT_DIR = "outputs/reports"

def load_and_merge():
    data = pd.read_csv(DATA_PATH, index_col=0)
    labels = pd.read_csv(LABELS_PATH, index_col=0)

    assert data.shape[0] == labels.shape[0], "Mismatched sample counts between data and labels"

    merged = data.join(labels, how="inner")
    print(f"Data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Merged shape: {merged.shape}")
    return data, labels, merged

def check_missing_and_variance(data):
    n_missing = data.isnull().sum().sum()
    zero_var_genes = data.columns[data.var(axis=0) == 0]
    print(f"Total missing values: {n_missing}")
    print(f"Zero-variance genes: {len(zero_var_genes)} out of {data.shape[1]}")
    return n_missing, zero_var_genes

def plot_class_distribution(labels):
    plt.figure(figsize=(7, 5))
    order = labels['Class'].value_counts().index
    sns.countplot(x='Class', data=labels, order=order)
    plt.title("Class Distribution: Cancer Types")
    plt.xlabel("Cancer Type")
    plt.ylabel("Sample Count")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/class_distribution.png", dpi=150)
    plt.close()
    print(f"Saved class distribution plot to {FIG_DIR}/class_distribution.png")

def plot_expression_summary(data):
    # Distribution of overall gene expression values (sanity check for scale/skew)
    sample_vals = data.values.flatten()
    sample_vals = np.random.choice(sample_vals, size=200_000, replace=False)  # subsample for speed

    plt.figure(figsize=(7, 5))
    sns.histplot(sample_vals, bins=100, kde=False)
    plt.title("Raw Gene Expression Value Distribution (sampled)")
    plt.xlabel("Expression value")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/expression_distribution_raw.png", dpi=150)
    plt.close()
    print(f"Saved raw expression distribution plot to {FIG_DIR}/expression_distribution_raw.png")

def write_eda_report(data, labels, n_missing, zero_var_genes):
    with open(f"{REPORT_DIR}/eda_summary.txt", "w") as f:
        f.write("=== EDA SUMMARY ===\n")
        f.write(f"Data shape: {data.shape}\n")
        f.write(f"Labels shape: {labels.shape}\n")
        f.write(f"Missing values: {n_missing}\n")
        f.write(f"Zero-variance genes: {len(zero_var_genes)}\n\n")
        f.write("Class distribution:\n")
        f.write(labels['Class'].value_counts().to_string())
        f.write("\n")
    print(f"Saved EDA summary to {REPORT_DIR}/eda_summary.txt")

if __name__ == "__main__":
    data, labels, merged = load_and_merge()
    n_missing, zero_var_genes = check_missing_and_variance(data)
    plot_class_distribution(labels)
    plot_expression_summary(data)
    write_eda_report(data, labels, n_missing, zero_var_genes)
    print("\nStage 3 complete ✅")
