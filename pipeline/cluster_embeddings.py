from transformers import EsmModel, EsmTokenizer
import torch
import numpy as np
from Bio import SeqIO
import os

# Set GPU for more efficient computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔌 Using device: {device}")

# Load ESM model & tokenizer
tokeniser = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
model.eval()

def embed_sequence(seq: str) -> np.ndarray:
    """
    Takes a string of amino acids and converts it into a numerical embedding.
    Returns a 1D numpy array with shape equal to the hidden size of the model.
    """
    tokens = tokeniser(seq, return_tensors="pt", add_special_tokens=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def embed_fasta(protein_family: str, fasta_path: str):
    """
    Reads a FASTA file, embeds each sequence with ESM2, and saves embeddings
    to ./data/initial_proteins/{protein_family}/{protein_family}_esm2_embeddings.npz
    """
    embeddings = []
    names = []

    print(f"📖 Reading sequences from {fasta_path}")
    for record in SeqIO.parse(fasta_path, "fasta"):
        emb = embed_sequence(str(record.seq))
        embeddings.append(emb)
        names.append(record.id)

    out_dir = f"./data/initial_proteins/{protein_family}/"
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, f"{protein_family}_esm2_embeddings.npz")
    np.savez_compressed(out_file, names=names, embeddings=np.vstack(embeddings))

    print(f"💾 Saved embeddings to {out_file}")

# Example usage
if __name__ == "__main__":
    protein_family = "your_family"  # e.g. "gyrase"
    fasta_path = f"./data/initial_proteins/{protein_family}/{protein_family}_fastas/all_sequences.fasta"
    embed_fasta(protein_family, fasta_path)
