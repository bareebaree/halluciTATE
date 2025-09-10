# pipeline/temberture_eval.py
# -*- coding: utf-8 -*-
"""
Run TemBERTure evaluation on a CSV of sequences.
Processes one replica at a time to avoid GPU OOM,
then merges results and computes average melt temp.
"""

import os
import sys
import argparse
import pandas as pd
import torch

# User should edit this path in the README instructions
temBERTure_path = "/mnt/c/Users/james/Masters_Degree/Thesis/protein_language_model_project/src/TemBERTure"

if os.path.exists(temBERTure_path):
    sys.path.append(temBERTure_path)
else:
    raise ImportError(
        f"TemBERTure path not found: {temBERTure_path}\n"
        "Please edit temBERTure_eval.py or set the correct path in your environment."
    )

from temBERTure import TemBERTure


def run_temBERTure(input_csv: str, results_dir: str = "./results") -> None:
    """
    Run TemBERTure evaluation on a CSV of sequences, one replica at a time.

    Parameters
    ----------
    input_csv : str
        Path to input CSV file. Must contain a 'sequence' column.
    results_dir : str, optional
        Base results directory. Outputs will go to results/temberture_results.
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    df = pd.read_csv(input_csv)
    if "sequence" not in df.columns:
        raise ValueError(f"Input file missing 'sequence' column: {input_csv}")

    # Prepare output directory
    output_dir = os.path.join(results_dir, "temberture_results")
    os.makedirs(output_dir, exist_ok=True)

    # Resolve adapter paths relative to temBERTure_path
    adapter_base = os.path.join(temBERTure_path, "temBERTure_TM")
    replica_paths = {
        "melt_temp_replica1": os.path.join(adapter_base, "replica1/"),
        "melt_temp_replica2": os.path.join(adapter_base, "replica2/"),
        "melt_temp_replica3": os.path.join(adapter_base, "replica3/"),
    }

    # Copy base dataframe so we can append predictions
    results_df = df.copy()

    # Run one replica at a time
    for col_name, adapter_path in replica_paths.items():
        print(f"Running predictions for {col_name} from {adapter_path}...")

        model = TemBERTure(
            adapter_path=adapter_path,
            device="cuda",
            batch_size=1,
            task="regression",
        )

        preds = []
        for seq in results_df["sequence"]:
            pred = model.predict(seq)[0]
            preds.append(pred)

        results_df[col_name] = preds

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

        # Save intermediate results
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        interim_out = os.path.join(output_dir, f"{base_name}_{col_name}.csv")
        results_df.to_csv(interim_out, index=False)
        print(f"Saved {col_name} predictions → {interim_out}")

    # Compute average melt temp
    results_df["avg_melt_temp"] = results_df[
        ["melt_temp_replica1", "melt_temp_replica2", "melt_temp_replica3"]
    ].mean(axis=1)

    # Save final combined CSV
    final_out = os.path.join(output_dir, f"{base_name}_melt_temps.csv")
    results_df.to_csv(final_out, index=False)
    print(f"Saved final combined predictions → {final_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TemBERTure evaluation on a CSV of sequences (one replica at a time).")
    parser.add_argument("input_csv", type=str, help="Path to input CSV (must contain a 'sequence' column)")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./results",
        help="Base results directory (default: ./results)",
    )

    args = parser.parse_args()
    run_temBERTure(args.input_csv, results_dir=args.results_dir)
