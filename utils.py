import os
import pickle

def load_pickle_file(file_path):
    """Loads a pickle file and returns the data."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_plot(fig, filename, filetype="fig", fmt="pdf"):
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