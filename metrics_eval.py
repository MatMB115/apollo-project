import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from utils import load_pickle_file, load_arguments, convert_to_dataframe, save_plot, dataframe_to_csv

def plot_macro_avg_roc(y_true, y_proba, classes, label):
    y_true_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)

    all_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])  # type: ignore
        mean_tpr += np.interp(all_fpr, fpr, tpr)

    mean_tpr /= n_classes
    auc_score = auc(all_fpr, mean_tpr)

    plt.plot(all_fpr, mean_tpr, label=f"{label} (Macro AUC = {auc_score:.3f})")

def find_best_scores(df):
    metrics = ["Accuracy", "F1-Score", "AUC", "Top_2"]
    categories = {"E": {}, "C": {}}

    for metric in metrics:
        for prefix in ["E-", "C-"]:
            col = f"{prefix}{metric}"
            if col in df.columns:
                best_value = df[col].max()
                best_k = df.loc[df[col].idxmax(), "K"]
                categories[prefix[0]][metric] = (best_k, best_value)

    return categories

def plot_best_scores_table(categories):
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    metrics = ["Accuracy", "F1-Score", "AUC", "Top_2"]
    headers = ["Metric", "Best K", "Best Value"]
    
    for idx, (key, title) in enumerate(zip(["E", "C"], ["Euclidean", "Cosine"])):
        data = [[metric, categories[key][metric][0], f"{categories[key][metric][1]:.5f}"] 
                for metric in metrics if metric in categories[key]]
        
        ax = axes[idx]
        ax.axis("tight")
        ax.axis("off")
        table = ax.table(cellText=data, colLabels=headers, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1, 2])
        
        ax.set_title(f"Best K and Metrics - {title}")
    
    plt.tight_layout()
    plt.show()

    save_plot(fig, "best_individual_metrics", filetype="tab", fmt="pdf")

def plot_table_metrics(df_results):
    df_results = df_results.reset_index()
    df_formatted = df_results.copy()
    for col in df_results.columns[1:]:
        df_formatted[col] = df_results[col].map(lambda x: f"{x:.5f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(cellText=df_formatted.values,
                     colLabels=df_formatted.columns,
                     cellLoc="center",
                     loc="center")

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(df_results.columns))))

    plt.title("K's Overall Average Metrics - Euclidean (E) vs Cosine (C)")

    save_plot(fig, "classification_average_metrics", filetype="tab", fmt="pdf")

    plt.show()

    plot_best_scores_table(find_best_scores(df_results))

def compute_metrics(results_dict):  
    knn_results_euclidean = results_dict["euclidean"]
    knn_results_cosine = results_dict["cosine"]

    df_results = convert_to_dataframe(knn_results_euclidean, knn_results_cosine, top_k=results_dict["top_k"])
    print("Top-K Accuracy: ", results_dict["top_k"])
    print("\nClassification Results Summary (Average Scores)")
    print(df_results.to_string())

    dataframe_to_csv(df_results, "classification_individual_metrics.csv")

    best_k_euclidean = max(knn_results_euclidean.keys(), key=lambda k: np.mean(knn_results_euclidean[k]["auc"]))
    best_k_cosine = max(knn_results_cosine.keys(), key=lambda k: np.mean(knn_results_cosine[k]["auc"]))

    print(f"Best K-neighboor (Euclidean): {best_k_euclidean}")
    print(f"Best K-neighboor (Cosine): {best_k_cosine}")

    y_true_euclidean = np.concatenate(knn_results_euclidean[best_k_euclidean]["y_true"])
    y_proba_euclidean = np.concatenate(knn_results_euclidean[best_k_euclidean]["y_proba"])

    y_true_cosine = np.concatenate(knn_results_cosine[best_k_cosine]["y_true"])
    y_proba_cosine = np.concatenate(knn_results_cosine[best_k_cosine]["y_proba"])

    unique_classes = np.unique(y_true_euclidean)

    plt.figure(figsize=(12, 8))
    plot_macro_avg_roc(y_true_euclidean, y_proba_euclidean, unique_classes, label="Euclidean")
    plot_macro_avg_roc(y_true_cosine, y_proba_cosine, unique_classes, label="Cosine")

    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.5)")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Macro-Averaged ROC Curve - Euclidean vs Cosine")
    plt.legend(loc="lower right")
    plt.grid(True)

    save_plot(plt, "macro_avg_roc", filetype="fig", fmt="pdf")

    plt.show()

    plot_table_metrics(df_results)

def main():
    param_dict = {
        "path": {
            "type": str,
            "default": "knn_detailed_results.p",
            "help": "Full path where the detailed KNN results are stored."
        }
    }

    args = load_arguments("Script to evaluate metrics and generate ROC AUC curves", param_dict)
    results_dict = load_pickle_file(args.path)

    compute_metrics(results_dict)

if __name__ == "__main__":
    main()
