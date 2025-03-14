import argparse
import pickle
import numpy as np
import pandas as pd
import os

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def main():
    parser = argparse.ArgumentParser(
        description="Script to preprocess a .p (pickle) embeddings file."
    )
    parser.add_argument(
        "--path", 
        type=str, 
        default="mini_gm_public_v0.1.p",
        help=(
            "Full path (including the .p file) where the embeddings are located."
            "Default path is 'mini_gm_public_v0.1.p' from the current directory."
        )
    )
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print(f"File not found at: {args.path}")
        print("Check the path or place 'mini_gm_public_v0.1.p' in the current directory.")
        return
    
    raw_data_dict = load_pickle_file(args.path)

    print(raw_data_dict)


if __name__ == "__main__":
    main()  