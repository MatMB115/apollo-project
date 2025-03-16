import os
import numpy as np
import pandas as pd

from utils import load_pickle_file, save_as_pickle, load_arguments

def flatten_data(data_dict):
    flattened = []

    # syndrome_id -> subject_id -> image_id -> embedding
    for syndrome_id, subjects_dict in data_dict.items():
        for subject_id, images_dict in subjects_dict.items():
            for image_id, embedding in images_dict.items():
                flattened.append({
                    'syndrome_id': syndrome_id,
                    'subject_id': subject_id,
                    'image_id': image_id,
                    'embedding': embedding
                })
    
    return pd.DataFrame(flattened)

def validate_flattened_data(df):
    issues = []

    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        issues.append("There are missing records in the dataset.")

    df["embedding_size"] = df["embedding"].apply(len)
    wrong_dim = df[df["embedding_size"] != 320]
    if not wrong_dim.empty:
        issues.append(f"{len(wrong_dim)} records have incorrect embedding dimensions.")

    valid_df = df[(df["embedding_size"] == 320) & df[["syndrome_id", "subject_id", "image_id"]].notnull().all(axis=1)]
    invalid_df = df[~df.index.isin(valid_df.index)]

    return valid_df.drop(columns=["embedding_size"]), invalid_df, issues

def calculate_gini_coefficient(values):
    values_sorted = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    gini_coefficient = (2 * np.sum(index * values_sorted) / (n * np.sum(values_sorted))) - (n + 1) / n
    return gini_coefficient

def generate_dataframe_statistics(df):
    print(f"Total raw records: {len(df)}")
    print("\n--- DataFrame Statistics ---")

    unique_syndromes = df["syndrome_id"].nunique()
    print(f"Unique syndromes: {unique_syndromes}")

    print("\nDistribution (images per syndrome):")
    syndrome_counts = df["syndrome_id"].value_counts()
    print(syndrome_counts)

    print("\n--- Unique Subjects per Syndrome ---")
    subject_counts = df.groupby("syndrome_id")["subject_id"].nunique()
    print(subject_counts.sort_values(ascending=False))
    
    gini_images = calculate_gini_coefficient(syndrome_counts.values)
    gini_subjects = calculate_gini_coefficient(subject_counts.values)

    print("\n--- Gini Coefficients ---")
    print(f"Gini Coefficient for Images per Syndrome: {gini_images:.4f}")
    print(f"Gini Coefficient for Subjects per Syndrome: {gini_subjects:.4f}")

def main():
    param_dict = {
        "path": {
            "type": str,
            "default": "mini_gm_public_v0.1.p",
            "help": (
                "Full path (including the .p file) where the embeddings are located. "
                "Default path is 'mini_gm_public_v0.1.p' from the current directory."
            )
        }
    }
    args = load_arguments("Script to preprocess a .p (pickle) embeddings file.", param_dict)

    if not os.path.isfile(args.path):
        print(f"File not found at: {args.path}")
        print("Check the path or place 'mini_gm_public_v0.1.p' in the current directory.")
        return
    
    raw_data_dict = load_pickle_file(args.path)
    print(f"Data loaded from '{args.path}'")
    print(f"Number of diferents syndromes: {len(raw_data_dict)}")

    df = flatten_data(raw_data_dict)

    valid_df, invalid_df, issues = validate_flattened_data(df)
    if issues:
        print("\nIssues found during validation:")
        for issue in issues:
            print(issue)

    generate_dataframe_statistics(valid_df)

    print(f"\nValid records: {len(valid_df)}")
    print(f"Invalid records: {len(invalid_df)}")

    if not invalid_df.empty:
        invalid_df.to_csv("invalid_records.csv", index=False)
        print("Invalid records saved to 'invalid_records.csv'.")

    save_as_pickle(valid_df, "mini_gm_public_v0.1_processed.p")

if __name__ == "__main__":
    main()  