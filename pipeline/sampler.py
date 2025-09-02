"""
This is a custom sampler.py modified from the original EvoProtGrad one. It has the functionality to have custom logging for recording mutations.
"""

import os, time, csv
from typing import List, Tuple, Optional
import torch, numpy as np
import evo_prot_grad.common.utils as utils
import evo_prot_grad.common.tokenizers as tokenizers
from evo_prot_grad.experts.base_experts import Expert


class DirectedEvolution:
    """Directed evolution with gradient-based discrete MCMC, extended with CSV logging."""

    def __init__(self,
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
                 results_dir: str = "./results"):

        self.experts = experts
        self.parallel_chains = parallel_chains
        self.n_steps = n_steps
        self.max_mutations = max_mutations
        self.output = output
        self.preserved_regions = preserved_regions
        self.wt_protein = wt_protein
        self.wt_fasta = wt_fasta
        self.verbose = verbose
        self.random_seed = random_seed
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        # timestamped master log
        run_stamp = time.strftime("%Y%m%d-%H%M%S")
        self.master_log_path = os.path.join(
            self.results_dir, f"evoprotgrad_run_log_master_{run_stamp}.csv"
        )

        if random_seed is not None:
            utils.set_seed(random_seed)

        self.canonical_chain_tokenizer = tokenizers.OneHotTokenizer(
            alphabet=utils.CANONICAL_ALPHABET
        )

        if self.wt_protein:
            if ".fasta" in self.wt_protein:
                raise ValueError("Use `wt_fasta` argument instead of `wt_protein`")
            self.wtseq = " ".join(self.wt_protein) if " " not in self.wt_protein else self.wt_protein
        else:
            self.wtseq = utils.read_fasta(self.wt_fasta)

        self.reset()
        self.max_pas_path_length = 2

    def reset(self):
        if self.random_seed is not None:
            utils.set_seed(self.random_seed)
        self.chains = [self.wtseq] * self.parallel_chains
        self.chains_oh = self.canonical_chain_tokenizer(self.chains).to(
            self.experts[0].device
        )
        self.wt_oh = self.chains_oh[0]
        self.PoE_history, self.chains_oh_history = [], []
        for expert in self.experts:
            expert.init_wildtype(self.wtseq)

    # keep your _product_of_experts, _compute_gradients, _get_variants_and_scores here
    # keep your __call__ implementation here
    # just ensure that at the end of __call__ you write both:
    #   - master CSV (append, timestamped filename)
    #   - per-iteration CSV (fresh each time)
