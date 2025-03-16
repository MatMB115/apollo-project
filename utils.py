import os
import pickle
import argparse
import pandas as pd
import numpy as np

def load_pickle_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist. Please check the path.")
    
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def save_as_pickle(valid_df, filename):
    with open(filename, "wb") as f:
        pickle.dump(valid_df, f)

    print(f"Processed data saved as '{filename}'.")

def load_arguments(script_description, param_dict):
    parser = argparse.ArgumentParser(description=script_description)

    for param, config in param_dict.items():
        parser.add_argument(
            f"--{param}",
            type=config.get("type", str),
            default=config.get("default"),
            help=config.get("help", "No description avaliable."),
            choices=config.get("choices", None)
        )

    return parser.parse_args()

def save_plot(fig, filename, filetype="assets", fmt="pdf"):
    if filetype == "fig":
        folder = os.path.join("assets", "figures")
    elif filetype == "tab":
        folder = os.path.join("assets", "tables")
    else:
        folder = "assets"

    os.makedirs(folder, exist_ok=True)

    filepath = os.path.join(folder, f"{filename}.{fmt}")

    fig.savefig(filepath, format=fmt, bbox_inches='tight')
    print(f"Plot saved to {filepath}")

def convert_to_dataframe(knn_results_euclidean, knn_results_cosine, top_k):
    rows = []
    for k in knn_results_euclidean.keys():
        rows.append({
            "K": k,
            "E-Accuracy": np.mean(knn_results_euclidean[k]["accuracy"]),
            "E-F1-Score": np.mean(knn_results_euclidean[k]["f1"]),
            "E-AUC": np.mean(knn_results_euclidean[k]["auc"]),
            f"E-Top_{top_k}": np.mean(knn_results_euclidean[k]["top_k"]),
            "C-Accuracy": np.mean(knn_results_cosine[k]["accuracy"]),
            "C-F1-Score": np.mean(knn_results_cosine[k]["f1"]),
            "C-AUC": np.mean(knn_results_cosine[k]["auc"]),
            f"C-Top_{top_k}": np.mean(knn_results_cosine[k]["top_k"])
        })

    df_results = pd.DataFrame(rows).set_index("K")
    return df_results

def dataframe_to_csv(df, filename):
    folder = os.path.join("assets", "tables")

    os.makedirs(folder, exist_ok=True)

    df.to_csv(os.path.join(folder, filename))