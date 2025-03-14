import pickle

def load_pickle_file(file_path):
    """Loads a pickle file and returns the data."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
