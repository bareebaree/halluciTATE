"""
This code is modified from the EvoProtGrad project, https://github.com/NREL/EvoProtGrad/

Set to instantiate with the parameters in my paper.

"""


import evo_prot_grad
from experts.esm_expert import EsmExpert
import os
import glob
import pandas as pd
from collections import deque
from typing import List


class EvoProtGradPipeline:
    def __init__(self, protein_family: str,
                 results_dir: str = "./results/",
                 data_dir: str = "./data/iterations/"):
        self.protein_family = protein_family
        self.initial_fastas_dir = f'./data/initial_proteins/{protein_family}/{protein_family}_fastas/most_distant_sequences'
        self.data_dir = data_dir
        self.results_dir = results_dir
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        # Initialise expert
        self.esm_expert = EsmExpert(
            scoring_strategy='pseudolikelihood_ratio',
            temperature=1.0
        )

    def get_latest_iteration_log(self, iteration_number: int) -> str:
        """Return the latest log CSV for a given iteration."""
        pattern = os.path.join(self.results_dir, f"evoprotgrad_run_log_iter{iteration_number}_*.csv")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(f"No log file found for iteration {iteration_number}")
        return max(matches, key=os.path.getmtime)

    def extract_best_variant_to_fasta(self, log_csv_path: str, output_fasta_path: str, iteration: int):
        """Extract the best variant from a CSV log and write it as FASTA."""
        df = pd.read_csv(log_csv_path)
        if df.empty:
            raise ValueError(f"Log {log_csv_path} is empty.")
        best_row = df.loc[df['score'].idxmax()]
        best_seq = best_row['sequence']
        best_score = best_row['score']
        best_chain = best_row['chain_idx']
        with open(output_fasta_path, 'w') as f:
            f.write(f">iteration_{iteration}_chain{best_chain}_score{best_score:.4f}\n")
            f.write(best_seq + "\n")
        print(f"✅ Wrote best variant from {log_csv_path} to {output_fasta_path}")

    def run_initial_iteration(self, fasta_path: str, pdb_id: str) -> str:
        """Run the initial DirectedEvolution iteration."""
        evo_init = evo_prot_grad.DirectedEvolution(
            wt_fasta=fasta_path,
            output="all",
            experts=[self.esm_expert],
            parallel_chains=4,
            n_steps=100,
            max_mutations=8,
            verbose=True,
            results_dir=self.results_dir,
        )
        evo_init.pdb_id = pdb_id
        evo_init(iteration_number=0)

        log_path = self.get_latest_iteration_log(iteration_number=0)
        init_fasta = os.path.join(self.data_dir, f"{pdb_id}_iteration_0.fasta")
        self.extract_best_variant_to_fasta(log_path, init_fasta, iteration=0)
        return init_fasta

    def run_evolution_loop(self, pdb_id: str, wt_fasta: str):
        """Iteratively run DirectedEvolution until stopping criteria is met."""
        iteration = 1
        recent_best_scores = deque(maxlen=10)

        while True:
            print(f"\n🧬 Running evo iteration {iteration} for {pdb_id}...")
            evo = evo_prot_grad.DirectedEvolution(
                wt_fasta=wt_fasta,
                output="all",
                experts=[self.esm_expert],
                parallel_chains=4,
                n_steps=50,
                max_mutations=4,
                verbose=True,
                results_dir=self.results_dir,
            )
            evo.pdb_id = pdb_id
            evo(iteration_number=iteration)

            log_path = self.get_latest_iteration_log(iteration)
            fasta_out = os.path.join(self.data_dir, f"{pdb_id}_iteration_{iteration}.fasta")
            self.extract_best_variant_to_fasta(log_path, fasta_out, iteration)

            df = pd.read_csv(log_path)
            best_score = df["score"].max()
            recent_best_scores.append(best_score)

            if len(recent_best_scores) == 10:
                avg_score = sum(recent_best_scores) / 10
                print(f"📊 Average best score over last 10 iterations: {avg_score:.4f}")
                if avg_score <= 0.01:
                    print(f"\n🛑 Stopping: Average best score over last 10 runs is ≤ 0.01")
                    break

            wt_fasta = fasta_out
            iteration += 1

        print(f"\n✅ Completed evolution for {pdb_id}")

    def run_pipeline(self):
        """Main entrypoint for running the pipeline on all initial FASTA files."""
        fasta_files = glob.glob(os.path.join(self.initial_fastas_dir, "*.fasta"))
        if not fasta_files:
            raise FileNotFoundError(f"No FASTA files found in {self.initial_fastas_dir}")

        for fasta_path in fasta_files:
            pdb_id = os.path.basename(fasta_path)[:4]
            print(f"\n🚀 Starting evolution for {pdb_id}...")
            init_fasta = self.run_initial_iteration(fasta_path, pdb_id)
            self.run_evolution_loop(pdb_id, init_fasta)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EvoProtGrad pipeline.")
    parser.add_argument("--protein_family", required=True, help="Protein family name")
    args = parser.parse_args()

    pipeline = EvoProtGradPipeline(protein_family=args.protein_family)
    pipeline.run_pipeline()
