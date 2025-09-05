# pipeline/__main__.py

import argparse
from .embed_sequences import embed_family
from .cluster_embeddings import run_cluster_and_analysis
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
    parser.add_argument("--k", type=int, default=10, help="Number of clusters (for clustering stage)")
    args = parser.parse_args()

    protein_family = args.family

    if args.stage in ["embed", "all"]:
        embed_family(protein_family)

    if args.stage in ["cluster", "all"]:
        run_cluster_and_analysis(protein_family, k=args.k)

    if args.stage in ["evolve", "all"]:
        evo_script = os.path.join("pipeline", "initialise_evo_prot_grad.py")
        subprocess.run(["python", evo_script, "--family", protein_family], check=True)

if __name__ == "__main__":
    main()
