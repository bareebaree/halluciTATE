import re
from typing import Optional, Tuple

_PDB_STEM = re.compile(r"^[0-9][A-Za-z0-9]{3}(?:_[A-Za-z0-9]{1,4})?$")

def is_pdb_stem(stem: str) -> bool:
    """
    Return True if a filename stem matches strict PDB conventions.

    Parameters
    ----------
    stem : str
        Filename stem without extension.

    Returns
    -------
    bool
        True if stem matches ^[0-9][A-Za-z0-9]{3}(?:_[A-Za-z0-9]{1,4})?$
    """
    return bool(_PDB_STEM.match(stem))


def parse_pdb_stem(stem: str) -> Tuple[str, Optional[str]]:
    """
    Parse a strict PDB filename stem into PDB ID and optional chain.

    Parameters
    ----------
    stem : str
        Filename stem (e.g., '1abc' or '1abc_A').

    Returns
    -------
    tuple
        (pdb_id, chain) where chain may be None.

    Raises
    ------
    ValueError
        If the stem is not a valid strict PDB stem.
    """
    if not is_pdb_stem(stem):
        raise ValueError(f"Invalid PDB stem: {stem!r}")
    if "_" in stem:
        pid, chain = stem.split("_", 1)
        return pid.lower(), chain.upper()
    return stem.lower(), None
