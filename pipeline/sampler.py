import os
import time
import csv
from datetime import datetime
from typing import List, Tuple, Optional

import torch
import numpy as np
import pandas as pd 
from .experts.esm_expert import EsmExpert
from evo_prot_grad.experts.base_experts import Expert
import evo_prot_grad.common.utils as utils
import evo_prot_grad.common.tokenizers as tokenizers
# pipeline/initialise_evo_prot_grad.py



class DirectedEvolution:
    """
    Plug-and-play directed evolution with gradient-based discrete MCMC,
    extended with CSV logging that matches the pipeline's expectations.
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
    ):
        """
        Initialise the sampler and logging.

        Parameters
        ----------
        experts : List[Expert]
            List of scoring experts.
        parallel_chains : int
            Number of parallel chains.
        n_steps : int
            Number of MCMC steps per call.
        max_mutations : int
            Maximum Hamming distance from wild type; set -1 to disable.
        output : str
            One of {'best','last','all'} for return values.
        preserved_regions : Optional[List[Tuple[int, int]]]
            Regions [start, end] that must not mutate (inclusive of start, end).
        wt_protein : Optional[str]
            Wild-type sequence string. Provide either this or wt_fasta.
        wt_fasta : Optional[str]
            FASTA path for wild-type. Provide either this or wt_protein.
        verbose : bool
            Whether to print progress.
        random_seed : Optional[int]
            Random seed for reproducibility.
        results_dir : str
            Directory for CSV outputs. Created if missing.

        Raises
        ------
        ValueError
            On invalid arguments.
        """
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
        os.makedirs(self.results_dir, exist_ok=True)

        if self.n_steps < 1:
            raise ValueError("`n_steps` must be >= 1")
        if not (self.wt_protein is not None or self.wt_fasta is not None):
            raise ValueError("Must provide one of `wt_protein` or `wt_fasta`")
        if self.output not in {"best", "last", "all"}:
            raise ValueError("`output` must be one of 'best', 'last' or 'all'")
        if len(self.experts) < 1:
            raise ValueError("Must provide at least one expert")
        if self.preserved_regions is not None:
            for start, end in self.preserved_regions:
                if end - start < 0:
                    raise ValueError("Preserved regions must be at least 1 amino acid long")

        if random_seed is not None:
            utils.set_seed(random_seed)

        self.canonical_chain_tokenizer = tokenizers.OneHotTokenizer(
            alphabet=utils.CANONICAL_ALPHABET
        )

        if self.wt_protein is not None:
            if ".fasta" in self.wt_protein:
                raise ValueError(
                    "Did you mean to use the `wt_fasta` argument instead of `wt_protein`?"
                )
            self.wtseq = self.wt_protein
            if " " not in self.wtseq:
                self.wtseq = " ".join(self.wtseq)
        else:
            self.wtseq = utils.read_fasta(self.wt_fasta)

        if self.verbose:
            print(f">Wildtype sequence: {self.wtseq}")

        # Logging artefacts
        run_stamp = time.strftime("%Y%m%d-%H%M%S")
        self.master_log_path = os.path.join(
            self.results_dir, f"evoprotgrad_run_log_master_{run_stamp}.csv"
        )
        self.pdb_id: Optional[str] = None

        self.reset()
        self.max_pas_path_length = 2

    def reset(self) -> None:
        """
        Reset internal state for a fresh run of chains.
        """
        if self.random_seed is not None:
            utils.set_seed(self.random_seed)

        self.chains = [self.wtseq] * self.parallel_chains
        self.chains_oh = self.canonical_chain_tokenizer(self.chains).to(self.device)
        self.wt_oh = self.chains_oh[0]
        self.PoE_history = []
        self.chains_oh_history = []

        for expert in self.experts:
            expert.init_wildtype(self.wtseq)

    def _product_of_experts(self, inputs: List[str]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Compute product-of-experts scores.

        Parameters
        ----------
        inputs : List[str]
            Protein sequences, length = parallel_chains.

        Returns
        -------
        ohs : List[torch.Tensor]
            One-hot inputs per expert.
        PoE : torch.Tensor
            Sum of expert scores, shape (parallel_chains,).
        """
        ohs = []
        scores = []
        for expert in self.experts:
            oh, score = expert(inputs)
            ohs.append(oh)
            scores.append(expert.temperature * score)
        return ohs, torch.stack(scores, dim=0).sum(dim=0)

    def _compute_gradients(self, ohs: List[torch.Tensor], PoE: torch.Tensor) -> torch.Tensor:
        """
        Compute gradients of PoE w.r.t. one-hot encodings, canonicalised per expert.

        Parameters
        ----------
        ohs : List[torch.Tensor]
            One-hot inputs, each of shape [n_chains, seq_len, vocab_size].
        PoE : torch.Tensor
            Scores, shape [n_chains].

        Returns
        -------
        torch.Tensor
            Gradients, shape [n_chains, seq_len, vocab_size].
        """
        oh_grads = torch.autograd.grad(PoE.sum(), ohs)

        summed_grads = []
        for expert, oh_grad in zip(self.experts, oh_grads):
            if oh_grad.shape[1] == self.chains_oh.shape[1] + 2:
                oh_grad = oh_grad[:, 1:-1]
            summed_grads.append(oh_grad @ expert.expert_to_canonical_order)

        return torch.stack(summed_grads, dim=0).sum(dim=0)

    def _get_variants_and_scores(self) -> Tuple[List[str], np.ndarray]:
        """
        Prepare return values per the selected `output` mode.

        Returns
        -------
        variants : List[str]
            Decoded sequences for the selected mode.
        scores : np.ndarray
            Corresponding scores.
        """
        if self.output == "last":
            variants = self.canonical_chain_tokenizer.decode(self.chains_oh_history[-1])
            scores = self.PoE_history[-1].numpy()
        elif self.output == "all":
            variants = []
            for i in range(len(self.chains_oh_history)):
                variants.append(self.canonical_chain_tokenizer.decode(self.chains_oh_history[i]))
            scores = torch.stack(self.PoE_history).numpy()
        else:
            best_idxs = torch.stack(self.PoE_history).argmax(0)
            chains_oh_history = torch.stack(self.chains_oh_history)
            variants = self.canonical_chain_tokenizer.decode(
                torch.stack([chains_oh_history[best_idxs[i], i] for i in range(self.parallel_chains)])
            )
            scores = torch.stack(
                [self.PoE_history[best_idxs[i]][i] for i in range(self.parallel_chains)]
            ).numpy()
        return variants, scores

    def _iter_log_path(self, iteration_number: int) -> str:
        """
        Build the per-iteration CSV path.

        Parameters
        ----------
        iteration_number : int
            Iteration index.

        Returns
        -------
        str
            CSV path for this iteration.
        """
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if self.pdb_id:
            fname = f"evoprotgrad_{self.pdb_id}_iter{iteration_number}_{stamp}.csv"
        else:
            fname = f"evoprotgrad_run_log_iter{iteration_number}_{stamp}.csv"
        return os.path.join(self.results_dir, fname)

    def _ensure_master_header(self) -> None:
        """
        Ensure the master CSV has its header row.
        """
        if not os.path.exists(self.master_log_path):
            with open(self.master_log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["timestamp", "iteration", "pdb_id", "chain_idx", "score", "results_csv"]
                )

    def __call__(self, iteration_number: int = 0) -> Tuple[List[str], np.ndarray]:
        """
        Run one MCMC round and write logs.

        Parameters
        ----------
        iteration_number : int, optional
            Iteration index used in filenames. Default is 0.

        Returns
        -------
        variants : List[str]
            Sequences per the selected `output` mode.
        scores : np.ndarray
            Scores corresponding to `variants`.
        """
        x_rank = len(self.chains_oh.shape) - 1
        seq_len = self.chains_oh.shape[-2]
        cur_chains_oh = self.chains_oh.clone()

        pos_mask = torch.zeros_like(cur_chains_oh).to(cur_chains_oh.device)
        if self.preserved_regions is not None:
            for min_pos, max_pos in self.preserved_regions:
                pos_mask[:, min_pos : max_pos + 1] = 1
        pos_mask = pos_mask.bool()
        pos_mask = pos_mask.reshape(self.parallel_chains, -1)

        for i in range(self.n_steps):
            U = torch.randint(1, 2 * self.max_pas_path_length, size=(self.parallel_chains, 1))
            max_u = int(torch.max(U).item())
            u_mask = torch.arange(max_u).expand(self.parallel_chains, max_u) < U
            u_mask = u_mask.float().to(cur_chains_oh.device)

            onehot_idx = []
            traj_list = []
            forward_categoricals = []

            ohs, PoE = self._product_of_experts(self.chains)
            grad_x = self._compute_gradients(ohs, PoE)

            with torch.no_grad():
                for step in range(max_u):
                    score_change = grad_x - (grad_x * cur_chains_oh).sum(-1).unsqueeze(-1)
                    traj_list.append(cur_chains_oh)
                    approx_forward_expert_change = score_change.reshape(self.parallel_chains, -1) / 2

                    if self.max_mutations > 0:
                        dist = utils.mut_distance(cur_chains_oh, self.wt_oh)
                        mask_flag = (dist == self.max_mutations).bool()
                        mask_flag = mask_flag.reshape(self.parallel_chains)
                        mask = utils.mutation_mask(cur_chains_oh, self.wt_oh)
                        mask = mask.reshape(self.parallel_chains, -1)
                        mask[~mask_flag] = False
                        approx_forward_expert_change[mask] = -np.inf

                    approx_forward_expert_change[pos_mask] = -np.inf

                    cd_forward = torch.distributions.one_hot_categorical.OneHotCategorical(
                        probs=utils.safe_logits_to_probs(approx_forward_expert_change)
                    )
                    forward_categoricals.append(cd_forward)
                    changes_all = cd_forward.sample((1,)).squeeze(0)
                    onehot_idx.append(changes_all)
                    changes_all = changes_all.view(self.parallel_chains, seq_len, -1)
                    row_select = changes_all.sum(-1).unsqueeze(-1)
                    new_x = cur_chains_oh * (1.0 - row_select) + changes_all
                    cur_u_mask = u_mask[:, step].unsqueeze(-1).unsqueeze(-1)
                    cur_chains_oh = cur_u_mask * new_x + (1 - cur_u_mask) * cur_chains_oh

                y = cur_chains_oh

            y_strs = self.canonical_chain_tokenizer.decode(y)
            ohs, proposed_PoE = self._product_of_experts(y_strs)
            grad_y = self._compute_gradients(ohs, proposed_PoE)
            grad_y = grad_y.detach()

            with torch.no_grad():
                traj_list.append(y)
                traj = torch.stack(traj_list[1:], dim=1)
                reverse_score_change = grad_y.unsqueeze(1) - (grad_y.unsqueeze(1) * traj).sum(-1).unsqueeze(-1)
                reverse_score_change = reverse_score_change.reshape(self.parallel_chains, max_u, -1) / 2.0
                log_ratio = 0
                for idx in range(len(onehot_idx)):
                    cd_reverse = torch.distributions.one_hot_categorical.OneHotCategorical(
                        probs=utils.safe_logits_to_probs(reverse_score_change[:, idx])
                    )
                    log_ratio += u_mask[:, idx] * (
                        cd_reverse.log_prob(onehot_idx[idx]) - forward_categoricals[idx].log_prob(onehot_idx[idx])
                    )

                m_term = proposed_PoE - PoE
                log_acc = m_term + log_ratio
                accepted = (log_acc.exp() >= torch.rand_like(log_acc)).float().view(-1, *([1] * x_rank))
                cur_chains_oh = y * accepted + (1.0 - accepted) * cur_chains_oh

            self.chains_oh = cur_chains_oh
            self.chains = self.canonical_chain_tokenizer.decode(cur_chains_oh)
            self.chains_oh_history.append(cur_chains_oh.clone().detach().cpu())
            PoE = proposed_PoE * accepted.squeeze() + PoE * (1.0 - accepted.squeeze())
            self.PoE_history.append(PoE.clone().detach().cpu())

            if self.verbose:
                x_strs = self.canonical_chain_tokenizer.decode(cur_chains_oh)
                for idx, variant in enumerate(x_strs):
                    print(f'>step {i}, chain {idx}, acceptance rate: {log_acc[idx].exp().item():.4f}, product of experts score: {PoE[idx]:.4f}')
                    # Remove spaces so seq and wtseq are comparable
                    v_clean = variant.replace(" ", "")
                    wt_clean = self.wtseq.replace(" ", "")
                    # Only call print_variant_in_color if lengths match
                    if len(v_clean) == len(wt_clean):
                        utils.print_variant_in_color(v_clean, wt_clean)
                    else:
                        print(f"[warn] skipping color-print: variant len {len(v_clean)} ≠ WT len {len(wt_clean)}")
            

            if self.max_mutations > 0:
                dist = utils.mut_distance(cur_chains_oh, self.wt_oh)
                mask_flag = (dist >= self.max_mutations).bool()
                mask_flag = mask_flag.reshape(self.parallel_chains)
                cur_chains_oh[mask_flag] = self.wt_oh

        # Prepare logging payload: final chain endpoints and scores
        final_sequences = self.chains
        final_scores = self.PoE_history[-1].numpy()
        ts = datetime.utcnow().isoformat()

        iter_csv = self._iter_log_path(iteration_number)
        with open(iter_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["chain_idx", "sequence", "score", "timestamp"])
            for i, (seq_i, score_i) in enumerate(zip(final_sequences, final_scores)):
                writer.writerow([i, seq_i, float(score_i), ts])

        self._ensure_master_header()
        with open(self.master_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            best_idx = int(np.argmax(final_scores))
            best_score = float(np.max(final_scores))
            writer.writerow([ts, iteration_number, self.pdb_id or "", best_idx, best_score, iter_csv])

        # Return in the original API format
        return self._get_variants_and_scores()
