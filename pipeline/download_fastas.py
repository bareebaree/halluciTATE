import os
import requests

"""
This script will programmatically download selected fasta files from the associated PDB files from PDB from the TSV file in the previous 
step. PDB is used as this method utilises fastas with PDB structures of experimentally determined structure.

"""
def download_fasta(pdb_id: str, output_dir: str) -> bool:
    """Downloads the fasta file for any given PDB ID."""
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id.upper()}"
    response = requests.get(url)
    if response.status_code == 200:
        os.makedirs(output_dir, exist_okay=True)
        with open(os.path.join(output_dir, f"{pdb_id}.fasta"), "w") as f:
            f.write(response.text)
        return True
    return False
