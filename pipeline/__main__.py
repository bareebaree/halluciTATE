# __main__.py

import argparse
from pipeline.embed import embed_fasta
from pipeline.cluster import run_cluster_and_analysis
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Protein pipeline")
    parser.add_argument("--family", type=str, required=True, help="Protein family name")
    parser.add_argument(
        "--stage",
        choices=["embed", "cluster", "evolve", "all"],
        default="all",
        help="Which pipeline stage to run",
    )
    parser.add_argument(
        "--k", type=int, default=10, help="Number of clusters (for clustering stage)"
    )
    args = parser.parse_args()

    protein_family = args.family

    # Stage 1: embeddings
    if args.stage in ["embed", "all"]:
        fasta_path = f"./data/initial_proteins/{protein_family}/{protein_family}_fastas/all_sequences.fasta"
        embed_fasta(protein_family, fasta_path)

    # Stage 2: clustering
    if args.stage in ["cluster", "all"]:
        run_cluster_and_analysis(protein_family, k=args.k)

    # Stage 3: evolution loop
    if args.stage in ["evolve", "all"]:
        evo_script = os.path.join("src", "initialise_evo_prot_grad.py")
        subprocess.run(["python", evo_script], check=True)

if __name__ == "__main__":
    main()
