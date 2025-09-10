# pipeline/fasta_to_csv.py
# -*- coding: utf-8 -*-
"""
Convert FASTA files into a long-format CSV, where each row is (pdb_id, sequence).
"""

import os
import argparse
import pandas as pd


def parse_fasta(filepath: str) -> list[str]:
    """
    Parse a FASTA file and return a list of sequences.

    Parameters
    ----------
    filepath : str
        Path to FASTA file.

    Returns
    -------
    list of str
        List of sequences contained in the FASTA file.
    """
    sequences = []
    current_seq = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq))

    return sequences


def fasta_dir_to_csv(protein_family: str, base_dir: str = "./data/initial_proteins") -> None:
    """
    Convert all FASTA files in the most_distant_sequences folder of a protein family
    into a long-format CSV. Each row corresponds to a (pdb_id, sequence) pair.

    Parameters
    ----------
    protein_family : str
        Protein family name.
    base_dir : str, optional
        Base directory containing protein family subfolders. Default is ./data/initial_proteins.
    """
    fasta_dir = os.path.join(
        base_dir, protein_family, f"{protein_family}_fastas", "most_distant_sequences"
    )
    if not os.path.isdir(fasta_dir):
        raise FileNotFoundError(f"FASTA directory not found: {fasta_dir}")

    rows = []

    for fname in sorted(os.listdir(fasta_dir)):
        if not fname.lower().endswith(".fasta"):
            continue
        pdb_id = fname[:4]
        fpath = os.path.join(fasta_dir, fname)
        sequences = parse_fasta(fpath)
        for seq in sequences:
            rows.append({"pdb_id": pdb_id, "sequence": seq})

    if not rows:
        raise ValueError(f"No FASTA files found in {fasta_dir}")

    df = pd.DataFrame(rows)
    output_csv = os.path.join(
        base_dir, protein_family, f"{protein_family}_most_distant_sequences.csv"
    )
    df.to_csv(output_csv, index=False)

    print(f"Saved long-format CSV to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FASTA files into a long-format CSV (pdb_id, sequence).")
    parser.add_argument("family", type=str, help="Protein family name")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./data/initial_proteins",
        help="Base directory containing protein family subfolders",
    )

    args = parser.parse_args()
    fasta_dir_to_csv(args.family, base_dir=args.base_dir)
