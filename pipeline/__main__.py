# pipeline/__main__.py

import argparse
import os
import sys
import subprocess
from .embed_sequences import embed_family
from .cluster_embeddings import run_cluster_and_analysis
from .load_structures import load_structure_ids
from .download_fastas import download_fasta


def download_family(protein_family: str, tsv_path: str) -> None:
    """
    Download FASTA files for a protein family using a TSV of PDB accessions.

    Parameters
    ----------
    protein_family : str
        Name of the protein family used to construct the output directory path.
    tsv_path : str
        Path to a TSV file with a column 'Accession' containing PDB IDs.
    """
    out_dir = (
        f"./data/initial_proteins/{protein_family}/"
        f"{protein_family}_fastas"
    )
    os.makedirs(out_dir, exist_ok=True)

    pdb_ids = load_structure_ids(tsv_path)
    if len(pdb_ids) == 0:
        raise ValueError(
            f"No PDB IDs found in TSV: {tsv_path}. "
            "Expected a column named 'Accession'."
        )

    downloaded = 0
    skipped = 0
    failed = 0

    for pdb_id in pdb_ids:
        dest_path = os.path.join(out_dir, f"{pdb_id.lower()}.fasta")
        if os.path.exists(dest_path):
            skipped += 1
            continue

        ok = download_fasta(pdb_id, out_dir)
        if ok:
            downloaded += 1
        else:
            failed += 1

    print(
        f"Downloader summary: downloaded={downloaded}, "
        f"skipped_existing={skipped}, failed={failed}. "
        f"Output dir: {out_dir}"
    )


def main() -> None:
    """
    Command-line entrypoint for the pipeline.
    """
    parser = argparse.ArgumentParser(description="Protein pipeline")
    parser.add_argument(
        "--family",
        type=str,
        required=True,
        help="Protein family name"
    )
    parser.add_argument(
        "--stage",
        choices=["download", "embed", "cluster", "evolve", "summarise", "temberture", "all"],
        default="all",
        help="Which pipeline stage to run"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of clusters (for clustering stage)"
    )
    parser.add_argument(
        "--tsv",
        type=str,
        help=(
            "TSV file with column 'Accession' for the download stage. "
            "Required for --stage download or --stage all."
        )
    )
    args = parser.parse_args()
    protein_family = args.family

    if args.stage in ["download", "all"]:
        if not args.tsv:
            raise ValueError(
                "Please provide --tsv pointing to a TSV with an 'Accession' "
                "column of PDB IDs for the download stage."
            )
        download_family(protein_family, args.tsv)

    if args.stage in ["embed", "all"]:
        embed_family(protein_family)

    if args.stage in ["cluster", "all"]:
        run_cluster_and_analysis(protein_family, k=args.k)

    if args.stage in ["evolve", "all"]:
        subprocess.run(
            [sys.executable, "-m", "pipeline.initialise_evo_prot_grad", "--family", protein_family],
            check=True
        )

    elif args.stage == "summarise":
    	from .select_best_scores import run_summary
    	run_summary(args.family)


    elif args.stage == "temberture":
        from .run_temberture import run_temBERTure
        run_temBERTure(protein_family)


if __name__ == "__main__":
    main()
