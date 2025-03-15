import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_pickle_file
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

def tsne_visualization(X, y, n_clusters=5, perplexity=30, random_state=100): 
    # sklearn change the name of arg n_iter to max_iter, so i use type: ignore
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000 # type: ignore 
    )

    X_tsne = tsne.fit_transform(X)
    # just for the sake of visualization, i will apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_tsne)

    plt.figure(figsize=(16, 10))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="viridis", alpha=0.7)
    plt.title("Visualization of Embeddings via t-SNE (Colored by Syndrome ID)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Syndrome ID", bbox_to_anchor=(1, 0.5), loc='center left')
    plt.show()

    plt.figure(figsize=(16, 10))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=cluster_labels, palette="tab10", alpha=0.7)
    plt.title("K-Means Clusters on t-SNE Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="K-Means Cluster", bbox_to_anchor=(1, 0.5), loc='center left')
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Script to visualize embeddings and apply K-Means clustering."
    )
    parser.add_argument(
        "--path", 
        type=str, 
        default="mini_gm_public_v0.1_processed.p",
        help=(
            "Full path (including the .p file) where the embeddings are located. "
            "Default path is 'mini_gm_public_v0.1_processed.p' from the current directory."
        )
    )
    parser.add_argument(
        "--n_clusters", 
        type=int, 
        default=10, 
        help="Number of clusters for K-Means. Default is 5."
    )
    args = parser.parse_args()

    df_embeddings = load_pickle_file(args.path)
    print(f"Data loaded from '{args.path}'")
    print(f"Number of records: {len(df_embeddings)}")

    X = np.vstack(df_embeddings["embedding"])
    y = df_embeddings["syndrome_id"].values

    tsne_visualization(X, y, n_clusters=args.n_clusters)

if __name__ == "__main__":
    main()