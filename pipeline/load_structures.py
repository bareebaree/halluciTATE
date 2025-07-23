import pandas as pd

def load_structure_ids(tsv_path: str) -> list:
    """Load PDB accession IDs from a TSV file.
       When using InterProt, selecting proteins with certain characteristics for their structural files will yield a TSV file
       with the PDB IDs for those proteins. This structure loads the accession IDs from and saves them onto a Pandas dataframe.
       This function requires the downloade TSV file as a parameter"""
       
    df = pd.read_csv(tsv_path, sep = "\t")
    pdb_ids = df['Accession'].dropna().astype(str).str.lower().unique()
    return pdb_ids