# main.py

import argparse
from pipeline.embed import embed_fasta

def main():
    parser = argparse.ArgumentParser(description="Protein ESM2 embedding pipeline")
    parser.add_argument("--family", type=str, required=True, help="Protein family name")
    args = parser.parse_args()

    protein_family = args.family
    fasta_path = f"./data/initial_proteins/{protein_family}/combined_{protein_family}s.fasta"
    output_path = f"./data/initial_proteins/{protein_family}/{protein_family}_esm2_embeddings.npz"

    embed_fasta(fasta_path, output_path)

if __name__ == "__main__":
    main()
