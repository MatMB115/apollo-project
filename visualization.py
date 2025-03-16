import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from utils import load_pickle_file, load_arguments, save_plot

def run_tsne_visualization(X, y, n_clusters=5, perplexity=30, random_state=115): 
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

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    sns.scatterplot(ax=axes[0], x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="tab10", alpha=0.7)
    axes[0].set_title("Visualization of Embeddings via t-SNE (Colored by Syndrome ID)")
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")
    axes[0].legend(title="Syndrome ID", bbox_to_anchor=(1, 0.5), loc='center left')
    
    sns.scatterplot(ax=axes[1], x=X_tsne[:, 0], y=X_tsne[:, 1], hue=cluster_labels, palette="tab10", alpha=0.7)
    axes[1].set_title("K-Means Clusters on t-SNE Embeddings")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    axes[1].legend(title="K-Means Cluster", bbox_to_anchor=(1, 0.5), loc='center left')
    
    plt.tight_layout()
    plt.show()

    save_plot(fig, "tsne_visualization", "fig")

def main():
    param_dict = {
        "path": {
            "type": str,
            "default": "mini_gm_public_v0.1_processed.p",
            "help": "Full path (including the .p file) where the processed embeddings are located."
        },
        "n_clusters": {
            "type": int,
            "default": 10,
            "help": "Number of clusters for K-Means. Default is 10."
        }
    }
    args = load_arguments("Script to visualize embeddings and apply K-Means clustering.", param_dict)

    df_embeddings = load_pickle_file(args.path)
    print(f"Data loaded from '{args.path}'")
    print(f"Number of records: {len(df_embeddings)}")

    X = np.vstack(df_embeddings["embedding"])
    y = df_embeddings["syndrome_id"].values

    run_tsne_visualization(X, y, n_clusters=args.n_clusters)

if __name__ == "__main__":
    main()
