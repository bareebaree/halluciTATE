# pipeline/temberture_eval.py
import os, sys
import pandas as pd

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

def run_temBERTure(protein_family: str, results_dir: str = "./results"):
    """
    Run TemBERTure evaluation on EvoProtGrad best-iteration outputs for a protein family.
    Saves per-sequence predictions and cumulative scores as CSVs.
    """
    input_csv = os.path.join(
        results_dir, f"{protein_family}_best_scores_per_iteration.csv"
    )
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Missing EvoProtGrad results CSV: {input_csv}")

    df = pd.read_csv(input_csv)

    # Initialise three replicas
    replicas = [
        TemBERTure(adapter_path="./temBERTure_TM/replica1/", device="cuda", batch_size=16, task="regression"),
        TemBERTure(adapter_path="./temBERTure_TM/replica2/", device="cuda", batch_size=16, task="regression"),
        TemBERTure(adapter_path="./temBERTure_TM/replica3/", device="cuda", batch_size=16, task="regression"),
    ]

    sequence_results = []
    cumulative_scores = {}

    for _, row in df.iterrows():
        pdb_id = row["pdb_id"]
        iteration = row["iteration"]
        sequence = row["sequence"]

        preds = [rep.predict(sequence)[0] for rep in replicas]
        avg_prediction = sum(preds) / len(preds)

        sequence_results.append({
            "pdb_id": pdb_id,
            "iteration": iteration,
            "sequence": sequence,
            "avg_prediction": avg_prediction,
        })

        cumulative_scores[pdb_id] = cumulative_scores.get(pdb_id, 0.0) + avg_prediction

    # Save outputs
    seq_outfile = os.path.join(results_dir, f"{protein_family}_temberture_sequence_predictions.csv")
    pd.DataFrame(sequence_results).to_csv(seq_outfile, index=False)

    cum_outfile = os.path.join(results_dir, f"{protein_family}_temberture_cumulative_scores.csv")
    pd.DataFrame([{"pdb_id": k, "cumulative_score": v} for k, v in cumulative_scores.items()]).to_csv(cum_outfile, index=False)

    print(f"Saved per-sequence predictions → {seq_outfile}")
    print(f"Saved cumulative scores → {cum_outfile}")
