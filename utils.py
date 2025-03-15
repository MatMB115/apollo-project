import os
import pickle

def load_pickle_file(file_path):
    """Loads a pickle file and returns the data."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_plot(fig, filename, folder="assets", format="pdf"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filepath = os.path.join(folder, f"{filename}.{format}")
    
    fig.savefig(filepath, format=format, bbox_inches='tight')
    print(f"Plot saved to {filepath}")