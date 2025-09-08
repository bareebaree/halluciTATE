# pipeline/initialise_evo_prot_grad.py

import argparse
import os
from collections import deque
from pathlib import Path
from typing import Optional, List

import pandas as pd

# Prefer local sampler; fall back if executed differently
try:
    from .sampler import DirectedEvolution
except Exception:
    from pipeline.sampler import DirectedEvolution  # type: ignore

from .experts.esm_expert import EsmExpert

RESULTS_DIR = Path("./results")
DATA_DIR = Path("./data/iterations")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_latest_iteration_log(
    iteration_number: int,
    stem: Optional[str] = None,
    results_dir: Path = RESULTS_DIR,
) -> str:
    """
    Find the newest EvoProtGrad CSV log for a given iteration.

    Parameters
    ----------
    iteration_number : int
        Iteration index (0-based).
    stem : Optional[str]
        Seed stem (e.g. '1abc_A'). If provided, the search will first
        look for files that contain the stem in the filename to avoid
        collisions when multiple seeds are run.
    results_dir : Path
        Directory where EvoProtGrad logs are written.

    Returns
    -------
    str
        Path to the newest matching CSV file.

    Raises
    ------
    FileNotFoundError
        If no matching log file is found.
    """
    candidates: List[Path] = []

    if stem:
        # Seed-scoped pattern (recommended for avoiding collisions)
        candidates.extend(results_dir.glob(f"evoprotgrad_{stem}_iter{iteration_number}_*.csv"))

    # Backward-compatible pattern (your original convention)
    if not candidates:
        candidates.extend(results_dir.glob(f"evoprotgrad_run_log_iter{iteration_number}_*.csv"))

    if not candidates:
        raise FileNotFoundError(
            f"No log file found for iteration {iteration_number} "
            f"(searched with stem={stem!r})"
        )

    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest.as_posix()


def extract_best_variant_to_fasta(
    log_csv_path: str,
    output_fasta_path: str,
    iteration: int,
) -> None:
    """
    Extract the best-scoring variant from a CSV log and write it to FASTA.

    Parameters
    ----------
    log_csv_path : str
        Path to the EvoProtGrad CSV log for a given iteration.
        The CSV must contain columns: 'sequence', 'score', 'chain_idx'.
    output_fasta_path : str
        Where to write the single-record FASTA file.
    iteration : int
        Iteration index (for the FASTA header).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the CSV is empty or required columns are missing.
    """
    df = pd.read_csv(log_csv_path)
    required = {"sequence", "score", "chain_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {log_csv_path}: {sorted(missing)}"
        )
    if df.empty:
        raise ValueError(f"Log {log_csv_path} is empty.")

    best_row = df.loc[df["score"].idxmax()]
    best_seq = str(best_row["sequence"])
    best_score = float(best_row["score"])
    best_chain = str(best_row["chain_idx"])

    with open(output_fasta_path, "w", encoding="utf-8") as handle:
        handle.write(f">iteration_{iteration}_chain{best_chain}_score{best_score:.4f}\n")
        handle.write(best_seq + "\n")

    print(f"Wrote best variant from {log_csv_path} to {output_fasta_path}")


def run_evolution(protein_family: str) -> None:
    """
    Run EvoProtGrad iterative evolution starting from seed FASTAs.

    Seeds are read from:
        ./data/initial_proteins/{family}/{family}_fastas/most_distant_sequences/*.fasta

    For each seed:
      1) Run an initial iteration (n_steps=100, max_mutations=8),
         extract best variant to ./data/iterations/{stem}_iteration_0.fasta.
      2) Run subsequent iterations (n_steps=50, max_mutations=4),
         each time using the previous best FASTA as the new wild-type.
      3) Stop when the average of the last 10 best scores ≤ 0.01.

    Parameters
    ----------
    protein_family : str
        Family name used to construct input and output paths.

    Returns
    -------
    None
    """
    initial_fastas_dir = (
        Path("./data/initial_proteins")
        / protein_family
        / f"{protein_family}_fastas"
        / "most_distant_sequences"
    )
    fasta_files = sorted(initial_fastas_dir.glob("*.fasta"))

    if not fasta_files:
        print(f"Warning: no FASTA files found in {initial_fastas_dir}")
        return

    esm_expert = EsmExpert(
        scoring_strategy="pseudolikelihood_ratio",
        temperature=1.0,
    )

    for initial_fasta in fasta_files:
        stem = initial_fasta.stem  # e.g. '1abc_A' or '2xyz'
        print(f"\nStarting evolution for {stem}...")

        # Initial iteration
        evo_init = DirectedEvolution(
            wt_fasta=str(initial_fasta),
            output="all",
            experts=[esm_expert],
            parallel_chains=4,
            n_steps=100,
            max_mutations=8,
            verbose=True,
            results_dir=RESULTS_DIR.as_posix(),
        )
        evo_init.pdb_id = stem
        evo_init(iteration_number=0)

        init_log_path = get_latest_iteration_log(iteration_number=0, stem=stem)
        init_fasta = DATA_DIR / f"{stem}_iteration_0.fasta"
        extract_best_variant_to_fasta(init_log_path, init_fasta.as_posix(), iteration=0)

        # Subsequent iterations
        iteration = 1
        wt_fasta = init_fasta.as_posix()
        recent_best_scores: deque = deque(maxlen=10)

        while True:
            print(f"\nRunning evo iteration {iteration} for {stem}...")

            evo = DirectedEvolution(
                wt_fasta=wt_fasta,
                output="all",
                experts=[esm_expert],
                parallel_chains=4,
                n_steps=50,
                max_mutations=4,
                verbose=True,
                results_dir=RESULTS_DIR.as_posix(),
            )
            evo.pdb_id = stem
            evo(iteration_number=iteration)

            log_path = get_latest_iteration_log(iteration_number=iteration, stem=stem)
            fasta_out = DATA_DIR / f"{stem}_iteration_{iteration}.fasta"
            extract_best_variant_to_fasta(log_path, fasta_out.as_posix(), iteration=iteration)

            df_iter = pd.read_csv(log_path)
            best_score = float(df_iter["score"].max())
            recent_best_scores.append(best_score)

            if len(recent_best_scores) == 10:
                avg_score = sum(recent_best_scores) / 10.0
                print(f"Average best score over last 10 iterations: {avg_score:.4f}")
                if avg_score <= 0.01:
                    print("Stopping: average best score threshold reached (≤ 0.01).")
                    break

            wt_fasta = fasta_out.as_posix()
            iteration += 1

        print(f"Completed evolution for {stem}.")


def main() -> None:
    """
    Parse command-line arguments and run the evolution routine.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="Run EvoProtGrad evolution")
    parser.add_argument(
        "--family",
        type=str,
        required=True,
        help="Protein family name (used for input/output paths)",
    )
    args = parser.parse_args()
    run_evolution(args.family)


if __name__ == "__main__":
    main()
