# pipeline/select_best_scores.py
# -*- coding: utf-8 -*-
"""
Summarise EvoProtGrad master logs by selecting the
highest-scoring sequence per iteration for each pdb_id.
Outputs both a CSV and a FASTA file, and also extracts the
final variant sequence (highest iteration per pdb_id).
"""

import os
import pandas as pd


def run_summary(family: str, results_dir: str = "./results") -> None:
    """
    Select the highest-scoring sequence from each iteration for every pdb_id.
    Save results as both CSV and FASTA. Also extract the final variant sequence
    (highest iteration per pdb_id) into a separate CSV.

    Parameters
    ----------
    family : str
        Protein family name (used to find the master log and name outputs).
    results_dir : str, optional
        Directory where master logs and output files are located. Default is ./results.
    """
    # Master log file for this family
    master_log = os.path.join(results_dir, f"evoprotgrad_run_log_master_{family}.csv")
    if not os.path.exists(master_log):
        raise FileNotFoundError(f"No master log found: {master_log}")

    # Read master log, drop any extra columns beyond the first 8
    df = pd.read_csv(master_log, usecols=range(8))

    # Force expected column names
    df.columns = [
        "timestamp",
        "iteration",
        "step",
        "chain_idx",
        "pdb_id",
        "sequence",
        "score",
        "acceptance_rate",
    ]

    # Clean up score + sequence
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["sequence"] = df["sequence"].astype(str).str.replace(" ", "", regex=False)

    # Drop invalid rows (no score or no sequence)
    df = df.dropna(subset=["score", "sequence"])
    if df.empty:
        raise ValueError(f"No valid rows found in master log: {master_log}")

    # Group by pdb_id and iteration, keep max score
    best_rows = df.loc[df.groupby(["pdb_id", "iteration"])["score"].idxmax()]

    # Save CSV
    output_csv = os.path.join(results_dir, f"{family}_best_scores_per_iteration.csv")
    best_rows.to_csv(output_csv, index=False)

    # Save FASTA
    fasta_path = os.path.splitext(output_csv)[0] + ".fasta"
    with open(fasta_path, "w", encoding="utf-8") as fasta_file:
        for _, row in best_rows.iterrows():
            pdb_id = row["pdb_id"]
            iteration = row["iteration"]
            score = row["score"]
            sequence = row["sequence"]
            header = f">{pdb_id}|iter={iteration}|score={score:.4f}"
            fasta_file.write(f"{header}\n{sequence}\n")

    # Extract final variant sequence per pdb_id (max iteration)
    final_variants = best_rows.loc[
        best_rows.groupby("pdb_id")["iteration"].idxmax()
    ].reset_index(drop=True)

    final_csv = os.path.join(results_dir, f"{family}_variant_sequences.csv")
    final_variants.to_csv(final_csv, index=False)

    print(f"Summary CSV saved to {output_csv}")
    print(f"FASTA file saved to {fasta_path}")
    print(f"Final variant CSV saved to {final_csv}")
