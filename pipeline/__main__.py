# main.py

import argparse
from pipeline.embed import embed_fasta
from pipeline.cluster import run_clustering_and_analysis

def main():
    parser = argparse.ArgumentParser(description="Protein pipeline")
    parser.add_argument("--family", type=str, required=True, help="Protein family name")
    parser.add_argument("--stage", choices=["embed", "cluster", "all"], default="all",
                        help="Which pipeline stage to run")

    args = parser.parse_args()
    protein_family = args.family

    if args.stage in ["embed", "all"]:
        fasta_path = f"./data/initial_proteins/{protein_family}/combined_{protein_family}s.fasta"
        embedding_output_path = f"./data/initial_proteins/{protein_family}/{protein_family}_esm2_embeddings.npz"
        embed_fasta(fasta_path, embedding_output_path)

    if args.stage in ["cluster", "all"]:
        run_clustering_and_analysis(protein_family)

if __name__ == "__main__":
    main()
