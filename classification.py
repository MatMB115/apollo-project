import numpy as np

from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, top_k_accuracy_score

from utils import load_pickle_file, load_arguments, save_as_pickle, convert_to_dataframe

def run_knn(X, y, kf, distance_metric="euclidean", k_neighboors=range(1, 16), top_k_acc=2):
    knn_results = {k: {"accuracy": [], "f1": [], "auc": [], "top_k": [], "y_true": [], "y_proba": []} for k in k_neighboors}

    for train_idx, test_idx in kf.split(X): # 10-fold cross-validation
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for k in k_neighboors:
            knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
            knn.fit(X_train, y_train)
            
            y_pred_proba = knn.predict_proba(X_test)
            y_pred = knn.predict(X_test)

            knn_results[k]["accuracy"].append(accuracy_score(y_test, y_pred))
            knn_results[k]["f1"].append(f1_score(y_test, y_pred, average="macro"))
            knn_results[k]["auc"].append(roc_auc_score(y_test, y_pred_proba, multi_class="ovr"))
            knn_results[k]["top_k"].append(top_k_accuracy_score(y_test, y_pred_proba, k=top_k_acc))

            knn_results[k]["y_true"].append(y_test)
            knn_results[k]["y_proba"].append(y_pred_proba)

    return knn_results

def main():
    param_dict = {
        "path": {
            "type": str,
            "default": "mini_gm_public_v0.1_processed.p",
            "help": "Full path (including the .p file) where the processed embeddings are located."
        },
        "top_k": {
            "type": int,
            "default": 2,
            "help": "Value of K for Top-K Accuracy (default: 2)."
        }
    }

    args = load_arguments("Script to do a classification task with K-Nearest Neighbors (KNN)", param_dict)
    df_embeddings = load_pickle_file(args.path)
    print(f"Data loaded from '{args.path}'")
    print(f"Number of records: {len(df_embeddings)}")
    print("Top-K Accuracy: ", args.top_k)

    X = np.vstack(df_embeddings["embedding"])
    y = df_embeddings["syndrome_id"].values

    kf = KFold(n_splits=10, shuffle=True, random_state=115)

    knn_results_euclidean = run_knn(X, y, kf, distance_metric="euclidean",top_k_acc=args.top_k)
    knn_results_cosine = run_knn(X, y, kf, distance_metric="cosine", top_k_acc=args.top_k)

    df_results = convert_to_dataframe(knn_results_euclidean, knn_results_cosine, args.top_k)

    print("\nClassification Results Summary (Average Scores)")
    print(df_results.to_string())

    results_dict = {
        "euclidean": knn_results_euclidean,
        "cosine": knn_results_cosine,
        "top_k": args.top_k
    }

    save_as_pickle(results_dict, "knn_detailed_results.p")

if __name__ == "__main__":
    main()
