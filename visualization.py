import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_pickle_file
from sklearn.manifold import TSNE

def tsne_visualization(X, y, perplexity=30, random_state=100): 
    # sklearn change the name of arg n_iter to max_iter, so i use type: ignore
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000 # type: ignore 
    )

    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(16, 10))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="viridis", alpha=0.7)
    plt.title("Visualization of Embeddings via t-SNE")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Syndrome ID", bbox_to_anchor=(1, 0.5), loc='center left')
    plt.show()

def main():
    parser = argparse.ArgumentParser(
    description="Script to visualize embeddings."
    )
    parser.add_argument(
        "--path", 
        type=str, 
        default="mini_gm_public_v0.1_processed.p",
        help=(
            "Full path (including the .p file) where the embeddings are located."
            "Default path is 'mini_gm_public_v0.1_processed.p' from the current directory."
        )
    )
    args = parser.parse_args()

    df_embeddings = load_pickle_file(args.path)
    print(f"Data loaded from '{args.path}'")
    print(f"Number of records: {len(df_embeddings)}")

    X = np.vstack(df_embeddings["embedding"])
    y = df_embeddings["syndrome_id"].values

    tsne_visualization(X, y)


if __name__ == "__main__":
    main()
