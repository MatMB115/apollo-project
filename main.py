import os
import argparse

from preprocessing import (
    flatten_data,
    validate_flattened_data,
    generate_dataframe_statistics,
)
from utils import (
    load_pickle_file,
    save_as_pickle,
)
from visualization import run_tsne_visualization
from classification import run_knn
from metrics_eval import compute_metrics

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold

def pipeline(args):
    """
    Executes the complete project pipeline:
    1. Preprocessing the original pickle file
    2. Visualizing embeddings (t-SNE)
    3. Classification (KNN) using Euclidean and Cosine distances
    4. Evaluation of metrics (AUC, F1, Top-K, etc.)
    """

    # ================================
    # 1) PREPROCESSING (load, flatten, validate and stats)
    # ================================
    print("\n--- [Step 1: Preprocessing] ---")
    if not os.path.isfile(args.input_pickle):
        raise FileNotFoundError(f"File '{args.input_pickle}' not found.")

    raw_data_dict = load_pickle_file(args.input_pickle)
    print(f"Loaded file with {len(raw_data_dict)} different syndromes.\n")

    df_raw = flatten_data(raw_data_dict)

    valid_df, invalid_df, issues = validate_flattened_data(df_raw)
    if issues:
        print("Issues found during validation:")
        for issue in issues:
            print(f"  - {issue}")

    generate_dataframe_statistics(valid_df)

    if not invalid_df.empty:
        invalid_df.to_csv("invalid_records.csv", index=False)
        print("\n[Info] Invalid records were saved to 'invalid_records.csv'.")

    processed_path = args.processed_pickle
    save_as_pickle(valid_df, processed_path)

    # ================================
    # 2) VISUALIZATION (t-SNE and K-Means)
    # ================================
    print("\n--- [Step 2: t-SNE Visualization] ---")
    df_embeddings = pd.read_pickle(processed_path)

    X = np.vstack(df_embeddings["embedding"])
    y = df_embeddings["syndrome_id"].values

    run_tsne_visualization(
        X,
        y,
        n_clusters=args.n_clusters,
        perplexity=args.tsne_perplexity,
        random_state=args.tsne_seed
    )

    # ================================
    # 3) CLASSIFICATION (KNN)
    # ================================
    print("\n--- [Step 3: KNN Classification] ---")

    kf = KFold(n_splits=10, shuffle=True, random_state=args.kfold_seed)

    k_neighbors_candidates = range(1, args.k_neighbors + 1)

    print("Training with Euclidean distance...")
    euclidean_results = run_knn(
        X, y, kf,
        distance_metric="euclidean",
        k_neighbors=k_neighbors_candidates,
        top_k_acc=args.top_k
    )

    print("Training with Cosine distance...")
    cosine_results = run_knn(
        X, y, kf,
        distance_metric="cosine",
        k_neighbors=k_neighbors_candidates,
        top_k_acc=args.top_k
    )

    results_dict = {
        "euclidean": euclidean_results,
        "cosine": cosine_results,
        "top_k": args.top_k
    }

    detailed_results_path = args.output_knn_results
    save_as_pickle(results_dict, detailed_results_path)
    print(f"Detailed KNN results saved to '{detailed_results_path}'.")

    # ================================
    # 4) METRICS EVALUATION
    # ================================
    print("\n--- [Step 4: Metrics Evaluation] ---")
    
    compute_metrics(results_dict)
    print("\n--- [Pipeline Completed] ---")


def main():
    parser = argparse.ArgumentParser(description="Complete Pipeline: Preprocessing, Visualization, Classification, and Metrics Evaluation.")

    parser.add_argument(
        "--input_pickle",
        type=str,
        default="mini_gm_public_v0.1.p",
        help="Path to the original pickle file (e.g., mini_gm_public_v0.1.p)."
    )
    parser.add_argument(
        "--processed_pickle",
        type=str,
        default="mini_gm_public_v0.1_processed.p",
        help="Output path to save the processed file."
    )
    parser.add_argument(
        "--output_knn_results",
        type=str,
        default="knn_detailed_results.p",
        help="Filename of the pickle file containing the detailed KNN results."
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=30.0,
        help="Perplexity parameter for t-SNE."
    )
    parser.add_argument(
        "--tsne_seed",
        type=int,
        default=115,
        help="Random state for t-SNE."
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=10,
        help="Number of clusters for K-Means (for visualization)."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=2,
        help="Value of K for Top-K Accuracy in KNN."
    )
    parser.add_argument(
        "--kfold_seed",
        type=int,
        default=115,
        help="Seed for KFold cross-validation."
    )
    parser.add_argument(
        "--k_neighbors",
        type=int,
        default=15,
        help="Value of neighbors in KNN."
    )

    args = parser.parse_args()
    pipeline(args)

if __name__ == "__main__":
    main()
