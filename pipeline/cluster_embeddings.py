import itertools
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

matplotlib.use("Agg")


def _parse_pdb_chain_from_header(header: str) -> Tuple[str, Optional[str]]:
    """
    Parse PDB ID and optional chain from a FASTA header.

    Parameters
    ----------
    header : str
        FASTA record.id or the full header string.

    Returns
    -------
    tuple
        (pdb_id_lower, chain_upper_or_None)

    Notes
    -----
    Handles common PDB-style headers such as:
      - 'pdb|1abc|A|...'
      - '1abc|CHAIN|A|...'
      - '1ABC_1|Chains A, B|...'
    Falls back to the first 4 characters if a cleaner token is not found.
    """
    h = header.strip()
    tokens = h.replace("pdb|", "").split("|")
    pid = None
    chain = None

    for tok in tokens:
        tok = tok.strip()
        if pid is None and len(tok) == 4 and tok[0].isdigit():
            pid = tok.lower()
            continue
        if chain is None and len(tok) == 1 and tok.isalnum():
            chain = tok.upper()

    if pid is None:
        pid = h[:4].lower()

    return pid, chain


def _write_single_chain_fasta(
    pdb_fasta_path: Path,
    out_path: Path,
    target_chain: Optional[str],
) -> None:
    """
    Write a single-record FASTA to out_path from a multi-record PDB FASTA.

    Parameters
    ----------
    pdb_fasta_path : Path
        Source FASTA that may contain multiple chains.
    out_path : Path
        Destination single-record FASTA path.
    target_chain : Optional[str]
        Chain identifier to select. If None, writes the first record.

    Raises
    ------
    FileNotFoundError
        If the source FASTA does not exist.
    ValueError
        If the source FASTA contains no records.
    """
    if not pdb_fasta_path.exists():
        raise FileNotFoundError(f"Missing source FASTA: {pdb_fasta_path}")

    records = list(SeqIO.parse(str(pdb_fasta_path), "fasta"))
    if not records:
        raise ValueError(f"No sequences in {pdb_fasta_path}")

    if target_chain is None:
        SeqIO.write(records[0], str(out_path), "fasta")
        print(f"Warning: no chain parsed; wrote first record from {pdb_fasta_path.name}")
        return

    for rec in records:
        toks = rec.id.replace("pdb|", "").split("|")
        if any(tok.strip().upper() == target_chain for tok in toks):
            SeqIO.write(rec, str(out_path), "fasta")
            return

    SeqIO.write(records[0], str(out_path), "fasta")
    print(
        f"Warning: chain {target_chain} not found in {pdb_fasta_path.name}; "
        f"wrote first record"
    )


def run_cluster_and_analysis(protein_family: str, k: int = 10) -> None:
    """
    Cluster ESM2 embeddings, save assignments and plots, and write per-seed FASTAs.

    Parameters
    ----------
    protein_family : str
        Name of the protein family used in directory paths.
    k : int, optional
        Number of KMeans clusters. Default is 10.

    Returns
    -------
    None

    Outputs
    -------
    - ./data/initial_proteins/{family}/{family}_cluster_assignments.csv
    - ./data/initial_proteins/{family}/closest_to_centroid.csv
      Columns: ID (seed stem), Cluster, Header (original FASTA header)
    - ./data/initial_proteins/{family}/tsne_top10_distant_clusters.png
    - ./data/initial_proteins/{family}/{family}_fastas/most_distant_sequences/{ID}.fasta
      Single-record seeds derived from either {pdb}.fasta or {pdb}_{CHAIN}.fasta.
    """
    data_path = (
        f"./data/initial_proteins/{protein_family}/"
        f"{protein_family}_esm2_embeddings.npz"
    )
    data = np.load(data_path, allow_pickle=True)
    names = data["names"]
    embeddings = data["embeddings"]

    print(f"Loaded {len(embeddings)} embeddings for {protein_family}")

    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    centroids = kmeans.cluster_centers_

    df = pd.DataFrame({"name": names, "cluster": labels})
    cluster_csv = (
        f"./data/initial_proteins/{protein_family}/"
        f"{protein_family}_cluster_assignments.csv"
    )
    df.to_csv(cluster_csv, index=False)
    print(f"Saved cluster assignments to {cluster_csv}")

    dist_matrix = pairwise_distances(centroids)
    pairs = list(itertools.combinations(range(k), 2))
    distances = sorted(
        [(i, j, float(dist_matrix[i, j])) for i, j in pairs],
        key=lambda x: x[2],
        reverse=True,
    )
    top_10 = distances[:10]

    print("Top 10 maximally distant cluster pairs:")
    for i, j, d in top_10:
        print(f"  Cluster {i} ↔ Cluster {j} → Distance: {d:.4f}")

    tsne = TSNE(n_components=2, random_state=42)
    reduced_all = tsne.fit_transform(np.vstack([embeddings, centroids]))
    reduced_embeddings = reduced_all[:-k]
    reduced_centroids = reduced_all[-k:]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        hue=labels,
        palette="tab10",
        s=40,
        legend="full",
        alpha=0.8,
    )
    plt.scatter(
        reduced_centroids[:, 0],
        reduced_centroids[:, 1],
        c="black",
        s=200,
        marker="X",
        label="Centroids",
    )
    for i, j, d in top_10:
        x_coords = [reduced_centroids[i, 0], reduced_centroids[j, 0]]
        y_coords = [reduced_centroids[i, 1], reduced_centroids[j, 1]]
        plt.plot(x_coords, y_coords, color="black", linestyle="--", alpha=0.6)
        mid_x, mid_y = float(np.mean(x_coords)), float(np.mean(y_coords))
        plt.text(mid_x, mid_y, f"{d:.1f}", fontsize=8, color="black")

    plt.title(
        f"t-SNE of {protein_family} embeddings\nTop 10 distant centroid pairs",
        fontsize=14,
    )
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    tsne_outfile = (
        f"./data/initial_proteins/{protein_family}/"
        f"tsne_top10_distant_clusters.png"
    )
    plt.savefig(tsne_outfile, dpi=300)
    print(f"Saved t-SNE plot to {tsne_outfile}")

    closest_sequences = []
    for cluster_id in range(k):
        indices = np.where(labels == cluster_id)[0]
        if indices.size == 0:
            continue
        cluster_embeddings = embeddings[indices]
        dists = np.linalg.norm(cluster_embeddings - centroids[cluster_id], axis=1)
        closest_idx_in_cluster = indices[int(np.argmin(dists))]

        header = str(names[closest_idx_in_cluster])
        pdb_id, chain = _parse_pdb_chain_from_header(header)
        out_stem = f"{pdb_id}_{chain}" if chain else pdb_id

        closest_sequences.append(
            {"ID": out_stem, "Cluster": int(cluster_id), "Header": header}
        )

    closest_df = pd.DataFrame(closest_sequences)
    closest_csv = f"./data/initial_proteins/{protein_family}/closest_to_centroid.csv"
    closest_df.to_csv(closest_csv, index=False)
    print(f"Saved closest-to-centroid list to {closest_csv}")

    fasta_src = Path(
        f"./data/initial_proteins/{protein_family}/{protein_family}_fastas"
    )
    fasta_dst = fasta_src / "most_distant_sequences"
    fasta_dst.mkdir(parents=True, exist_ok=True)

    for entry in closest_sequences:
        stem = entry["ID"]
        pdb_id = stem.split("_")[0]
        chain = stem.split("_")[1] if "_" in stem else None

        multifasta = fasta_src / f"{pdb_id}.fasta"
        out_path = fasta_dst / f"{stem}.fasta"

        if multifasta.exists():
            _write_single_chain_fasta(multifasta, out_path, chain)
            print(f"Wrote seed: {out_path.name}")
            continue

        per_chain = fasta_src / f"{stem}.fasta"
        if per_chain.exists():
            shutil.copy(per_chain, out_path)
            print(f"Copied seed: {out_path.name}")
            continue

        candidates = sorted(fasta_src.glob(f"{pdb_id}_*.fasta"))
        if candidates:
            shutil.copy(candidates[0], out_path)
            print(
                f"Warning: exact seed {stem}.fasta not found; used {candidates[0].name}"
            )
        else:
            print(f"Missing FASTA for: {stem}")

    print("Clustering and visualisation complete.")
