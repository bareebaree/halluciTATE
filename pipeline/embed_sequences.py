import os
import numpy as np
from Bio import SeqIO
import torch
from transformers import EsmModel, EsmTokenizer

_device = None
_tokeniser = None
_model = None

def _load_model():
    """
    If no device is chosen, this uses cuda if it is available, otherwise it uses cpu. 
    It then loads the 650M parameter ESM-2 tokeniser, then passes the model to the device, and runs .eval()
    """

    global _device, _tokeniser, _model
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" Using device: {_device}")
        _tokeniser = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        _model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(_device)
        _model.eval()
    return _device, _tokeniser, _model

def embed_sequence(seq: str) -> np.ndarray:
    """
    Unpacks, device, tokeniser variables from previous function.
    Encodes input tokens into pytorch tensors
    Moves each tensor in the k:v dictionary into the device
    Input tensors to model
    Get tensor of shape (batch_size, seq_length, hidden_dim) where hidden_dim is 1280 in ESM-2 650M, take mean across residues, created fixed size vector and move to np array on cpu
    
    """
    
    device, tokeniser, model = _load_model()
    tokens = tokeniser(seq, return_tensors="pt", add_special_tokens=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def embed_family(protein_family: str):
    """
    Embed all FASTA files in ./data/initial_proteins/{family}/{family}_fastas/
    Save a single NPZ with all embeddings.
    Opens every fasta with SeqIO then calls embed_sequence() on it, saving them all as a 2d array, with each row as a vector representing a sequence.
    """
    fasta_dir = f"./data/initial_proteins/{protein_family}/{protein_family}_fastas"
    out_file = f"./data/initial_proteins/{protein_family}/{protein_family}_esm2_embeddings.npz"
    embeddings, names = [], []

    if not os.path.exists(fasta_dir):
        raise FileNotFoundError(f"No FASTA directory found: {fasta_dir}")

    fasta_files = [f for f in os.listdir(fasta_dir) if f.endswith(".fasta")]
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found in: {fasta_dir}")

    print(f"📖 Reading {len(fasta_files)} FASTA files from {fasta_dir}")
    for fasta_file in fasta_files:
        path = os.path.join(fasta_dir, fasta_file)
        for record in SeqIO.parse(path, "fasta"):
            emb = embed_sequence(str(record.seq))
            embeddings.append(emb)
            names.append(record.id)

    np.savez_compressed(out_file, names=names, embeddings=np.vstack(embeddings))
    print(f"Saved embeddings to {out_file}")
