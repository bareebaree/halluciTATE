import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import itertools
import os
import shutil

# Allow matplotlib to work without GUI
matplotlib.use("Agg")

def run_cluster_and_analysis(protein_family: str, k: int = 10):
    """
    Runs K-means clustering on ESM2 embeddings of a given protein family,
    identifies maximally distant clusters, saves plots and CSVs,
    and copies representative FASTAs into most_distant_sequences/.

    Parameters
    ----------
    protein_family : str
        Name of the protein family (used for directory paths).
    k : int, optional
        Number of clusters to compute. Default = 10.
    """
    # Load embeddings
    data_path = f"./data/initial_proteins/{protein_family}/{protein_family}_esm2_embeddings.npz"
    data = np.load(data_path, allow_pickle=True)
    names = data["names"]
    embeddings = data["embeddings"]

    print(f"Loaded {len(embeddings)} embeddings for {protein_family}")

    # Run KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    # Save cluster assignments
    df = pd.DataFrame({"name": names, "cluster": labels})
    cluster_csv = f"./data/initial_proteins/{protein_family}/{protein_family}_cluster_assignments.csv"
    df.to_csv(cluster_csv, index=False)
    print(f"Saved cluster assignments to {cluster_csv}")

    # Compute pairwise centroid distances
    dist_matrix = pairwise_distances(centroids)
    pairs = list(itertools.combinations(range(k), 2))
    distances = sorted(
        [(i, j, dist_matrix[i, j]) for i, j in pairs],
        key=lambda x: x[2],
        reverse=True
    )
    top_10 = distances[:10]

    print("Top 10 maximally distant cluster pairs:")
    for i, j, d in top_10:
        print(f"  Cluster {i} ↔ Cluster {j} → Distance: {d:.4f}")

    # 2D t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42)
    reduced_all = tsne.fit_transform(np.vstack([embeddings, centroids]))
    reduced_embeddings = reduced_all[:-k]
    reduced_centroids = reduced_all[-k:]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1],
        hue=labels, palette="tab10", s=40, legend="full", alpha=0.8
    )
    plt.scatter(
        reduced_centroids[:, 0], reduced_centroids[:, 1],
        c="black", s=200, marker="X", label="Centroids"
    )
    for i, j, d in top_10:
        x_coords = [reduced_centroids[i, 0], reduced_centroids[j, 0]]
        y_coords = [reduced_centroids[i, 1], reduced_centroids[j, 1]]
        plt.plot(x_coords, y_coords, color="black", linestyle="--", alpha=0.6)
        mid_x, mid_y = np.mean(x_coords), np.mean(y_coords)
        plt.text(mid_x, mid_y, f"{d:.1f}", fontsize=8, color="black")

    plt.title(f"t-SNE of {protein_family} embeddings\nTop 10 distant centroid pairs", fontsize=14)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    tsne_outfile = f"./data/initial_proteins/{protein_family}/tsne_top10_distant_clusters.png"
    plt.savefig(tsne_outfile, dpi=300)
    print(f"Saved t-SNE plot to {tsne_outfile}")

    # Pick closest-to-centroid representatives
    closest_sequences = []
    for cluster_id in range(k):
        indices = np.where(labels == cluster_id)[0]
        cluster_embeddings = embeddings[indices]
        dists = np.linalg.norm(cluster_embeddings - centroids[cluster_id], axis=1)
        closest_idx_in_cluster = indices[np.argmin(dists)]
        raw_name = names[closest_idx_in_cluster]
        cleaned_name = raw_name[:4]
        closest_sequences.append({"PDB_ID": cleaned_name, "Cluster": cluster_id})

    closest_df = pd.DataFrame(closest_sequences)
    closest_csv = f"./data/initial_proteins/{protein_family}/closest_to_centroid.csv"
    closest_df.to_csv(closest_csv, index=False)
    print(f" Saved closest-to-centroid list to {closest_csv}")

    # Copy FASTA files for chosen representatives
    fasta_src = f"./data/initial_proteins/{protein_family}/{protein_family}_fastas"
    fasta_dst = os.path.join(fasta_src, "most_distant_sequences")
    os.makedirs(fasta_dst, exist_ok=True)

    for entry in closest_sequences:
        pdb_id = entry["PDB_ID"].lower()
        src_file = os.path.join(fasta_src, f"{pdb_id}.fasta")
        dst_file = os.path.join(fasta_dst, f"{pdb_id}.fasta")
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            print(f"Copied {pdb_id}.fasta → most_distant_sequences/")
        else:
            print(f"Missing FASTA for: {pdb_id}")

    print("Clustering + visualisation complete.")
