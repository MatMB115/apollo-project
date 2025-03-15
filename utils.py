import os
import pickle
import argparse

def load_pickle_file(file_path):
    """Loads a pickle file and returns the data."""
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