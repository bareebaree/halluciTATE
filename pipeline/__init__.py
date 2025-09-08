"""
Pipeline package for protein sequence embedding, clustering,
and directed evolution initialisation.
"""

from .embed_sequences import embed_family
from .cluster_embeddings import run_cluster_and_analysis

__all__ = [
    "embed_family",
    "run_cluster_and_analysis",
]
