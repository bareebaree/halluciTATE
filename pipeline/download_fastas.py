import os
from pathlib import Path
from typing import Optional
import requests
from Bio import SeqIO
from .utils_ids import is_pdb_stem

def download_fasta(pdb_id: str, output_dir: str) -> bool:
    """
    Download FASTA(s) for a PDB ID and write one file per chain.

    Parameters
    ----------
    pdb_id : str
        Four-character PDB accession. Case-insensitive.
    output_dir : str
        Destination directory.

    Returns
    -------
    bool
        True if at least one chain FASTA was written, else False.

    Notes
    -----
    Filenames are enforced as lowercase stems. Chains are uppercased
    in the filename (e.g., '1abc_A.fasta').
    """
    pid = pdb_id.strip().lower()
    if len(pid) != 4 or not pid[0].isdigit():
        raise ValueError(f"Invalid PDB ID: {pdb_id!r}")

    url = f"https://www.rcsb.org/fasta/entry/{pid.upper()}"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return False

    os.makedirs(output_dir, exist_ok=True)
    tmp = Path(output_dir) / f"{pid}.tmp.fasta"
    tmp.write_text(resp.text)

    written = 0
    for record in SeqIO.parse(str(tmp), "fasta"):
        chain: Optional[str] = None
        parts = record.id.split("|")
        # Best-effort PDB chain extraction; fallback to None.
        for p in parts:
            if len(p) in (1, 2, 3, 4) and p.isalnum():
                chain = p.upper()
        stem = pid if chain is None else f"{pid}_{chain}"
        if not is_pdb_stem(stem):
            # Enforce strict mode; refuse to write ambiguous names.
            continue
        out = Path(output_dir) / f"{stem}.fasta"
        SeqIO.write(record, str(out), "fasta")
        written += 1

    tmp.unlink(missing_ok=True)
    return written > 0
