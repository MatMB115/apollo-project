import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, top_k_accuracy_score

from utils import load_pickle_file, load_arguments, save_as_pickle

def run_knn(X, y, kf, distance_metric="euclidean", k_values=range(1, 16)):
    knn_results = {k: {"accuracy": [], "f1": [], "auc": [], "top_k": [], "y_true": [], "y_proba": []} for k in k_values}

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
            knn.fit(X_train, y_train)
            
            y_pred_proba = knn.predict_proba(X_test)
            y_pred = knn.predict(X_test)

            knn_results[k]["accuracy"].append(accuracy_score(y_test, y_pred))
            knn_results[k]["f1"].append(f1_score(y_test, y_pred, average="macro"))
            knn_results[k]["auc"].append(roc_auc_score(y_test, y_pred_proba, multi_class="ovr"))
            knn_results[k]["top_k"].append(top_k_accuracy_score(y_test, y_pred_proba, k=2))

            knn_results[k]["y_true"].append(y_test)
            knn_results[k]["y_proba"].append(y_pred_proba)

    return knn_results

def convert_to_dataframe(knn_results_euclidean, knn_results_cosine):
    rows = []
    for k in knn_results_euclidean.keys():
        rows.append({
            "K": k,
            "E-Accuracy": np.mean(knn_results_euclidean[k]["accuracy"]),
            "E-F1-Score": np.mean(knn_results_euclidean[k]["f1"]),
            "E-AUC": np.mean(knn_results_euclidean[k]["auc"]),
            "E-Top_2": np.mean(knn_results_euclidean[k]["top_k"]),
            "C-Accuracy": np.mean(knn_results_cosine[k]["accuracy"]),
            "C-F1-Score": np.mean(knn_results_cosine[k]["f1"]),
            "C-AUC": np.mean(knn_results_cosine[k]["auc"]),
            "C-Top_2": np.mean(knn_results_cosine[k]["top_k"])
        })
    
    df_results = pd.DataFrame(rows).set_index("K")
    return df_results

def main():
    param_dict = {
        "path": {
            "type": str,
            "default": "mini_gm_public_v0.1_processed.p",
            "help": "Full path (including the .p file) where the processed embeddings are located."
        }
    }

    args = load_arguments("Script to do a classification task with K-Nearest Neighbors (KNN)", param_dict)
    df_embeddings = load_pickle_file(args.path)
    print(f"Data loaded from '{args.path}'")
    print(f"Number of records: {len(df_embeddings)}")

    X = np.vstack(df_embeddings["embedding"])
    y = df_embeddings["syndrome_id"].values

    kf = KFold(n_splits=10, shuffle=True, random_state=115)

    knn_results_euclidean = run_knn(X, y, kf, distance_metric="euclidean")
    knn_results_cosine = run_knn(X, y, kf, distance_metric="cosine")

    df_results = convert_to_dataframe(knn_results_euclidean, knn_results_cosine)

    print("\nClassification Results Summary (Average Scores)")
    print(df_results.to_string())

    results_dict = {
        "euclidean": knn_results_euclidean,
        "cosine": knn_results_cosine,
        "metrics_summary": df_results
    }

    save_as_pickle(results_dict, "knn_detailed_results.p")

if __name__ == "__main__":
    main()
