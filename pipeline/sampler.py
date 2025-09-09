import os
import csv
from datetime import datetime
from typing import List, Tuple, Optional

import torch
import numpy as np
from evo_prot_grad.experts.base_experts import Expert
import evo_prot_grad.common.utils as utils
import evo_prot_grad.common.tokenizers as tokenizers


class DirectedEvolution:
    """
    Directed evolution with gradient-based discrete MCMC,
    extended with per-step master logging and per-iteration CSVs.
    """

    def __init__(
        self,
        experts: List[Expert],
        parallel_chains: int,
        n_steps: int,
        max_mutations: int,
        output: str = "last",
        preserved_regions: Optional[List[Tuple[int, int]]] = None,
        wt_protein: Optional[str] = None,
        wt_fasta: Optional[str] = None,
        verbose: bool = False,
        random_seed: Optional[int] = None,
        results_dir: str = "./results",
        family: Optional[str] = None,
    ):
        self.experts = experts
        self.parallel_chains = int(parallel_chains)
        self.n_steps = int(n_steps)
        self.max_mutations = int(max_mutations)
        self.output = output
        self.preserved_regions = preserved_regions
        self.wt_protein = wt_protein
        self.wt_fasta = wt_fasta
        self.verbose = verbose
        self.random_seed = random_seed
        self.device = self.experts[0].device
        self.results_dir = results_dir
        self.family = family or "unknown"

        os.makedirs(self.results_dir, exist_ok=True)

        if self.n_steps < 1:
            raise ValueError("`n_steps` must be >= 1")
        if not (self.wt_protein is not None or self.wt_fasta is not None):
            raise ValueError("Must provide one of `wt_protein` or `wt_fasta`")
        if self.output not in {"best", "last", "all"}:
            raise ValueError("`output` must be one of 'best', 'last', or 'all'")
        if len(self.experts) < 1:
            raise ValueError("Must provide at least one expert")
        if self.preserved_regions is not None:
            for start, end in self.preserved_regions:
                if end - start < 0:
                    raise ValueError("Preserved regions must be ≥ 1 aa long")

        if random_seed is not None:
            utils.set_seed(random_seed)

        self.canonical_chain_tokenizer = tokenizers.OneHotTokenizer(
            alphabet=utils.CANONICAL_ALPHABET
        )

        if self.wt_protein is not None:
            if ".fasta" in self.wt_protein:
                raise ValueError(
                    "Did you mean to use the `wt_fasta` argument instead?"
                )
            self.wtseq = self.wt_protein
            if " " not in self.wtseq:
                self.wtseq = " ".join(self.wtseq)
        else:
            self.wtseq = utils.read_fasta(self.wt_fasta)

        if self.verbose:
            print(f">Wildtype sequence: {self.wtseq}")

        # one timestamp for the entire run
        self.run_start_ts = datetime.utcnow().isoformat()

        # family-specific master log file
        self.master_log_path = os.path.join(
            self.results_dir, f"evoprotgrad_run_log_master_{self.family}.csv"
        )
        self._ensure_master_header()

        self.pdb_id: Optional[str] = None
        self.reset()
        self.max_pas_path_length = 2

    def _ensure_master_header(self) -> None:
        """Ensure the master CSV has a header."""
        if not os.path.exists(self.master_log_path):
            with open(self.master_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "iteration",
                        "step",
                        "chain_idx",
                        "pdb_id",
                        "sequence",
                        "score",
                        "acceptance_rate",
                    ]
                )

    def _iter_log_path(self, iteration_number: int) -> str:
        """Build the per-iteration CSV path."""
        stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        family_tag = f"{self.family}_" if self.family else ""
        if self.pdb_id:
            fname = f"evoprotgrad_{family_tag}{self.pdb_id}_iter{iteration_number}_{stamp}.csv"
        else:
            fname = f"evoprotgrad_{family_tag}run_log_iter{iteration_number}_{stamp}.csv"
        return os.path.join(self.results_dir, fname)

    # … keep the rest of the implementation identical …
