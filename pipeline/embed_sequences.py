"""
### OPTIONAL SCRIPT ###

This script is an optional argument, and is used for replicating the research step, if it is necessary to embed and cluster families to ensure they are distant.
If there is no need to select proteins by distance, then this can be skipped.

This script embeds a strings of amino acids into a numerical embedding sequence, so they can be clustered downstream.

This utilises the ESM tokeniser to do so, a pretrained model.

The output of this script is the embeddings of all protein sequences saved as a numpy NPZ file.



"""


from transformers import EsmModel, EsmTokenizer
import torch
import numpy as np
from Bio import SeqIO
import os


# Set GPU for more efficient computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔌 Using device: {device}")

# Load ESM model & tokeniser. Converts amino acid letters into a numerical tokens.
tokeniser = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

#load the respective ESM model for the tokeniser
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = model.to(device)

# Turn on evaluation mode
model.eval()

def embed_sequence(seq):
    """
    Takes a string of amino acids and converts into a numerical embedding sequence.
    
    Parameters: Sequence (str)
    
    Returns: An embedding: 1D numpy array with a shape equal to hidden size of model (numpy.ndarray)
    """
    tokens = tokeniser(seq, return_tensors="pt", add_special_tokens=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        output = model(**tokens)
    embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

def embed_fasta(fasta_path: str, embedding_output_path: str):
    """
    Read a fasta file, embed each sequence using ESM2, saves as an .npz file
    """
    
    embeddings = []
    names = []

    print(f"Reading sequences from {fasta_path}")
    for record in SeqIO.parse(fasta_path, "fasta"):
        emb = embed_sequence(str(record.seq))
        embeddings.append(emb)
        names.append(record.id)

# save
    os.makedirs(os.path.dirname(embedding_output_path), exist_ok=True)
    np.savez_compressed(embedding_output_path,
                        names=names, embeddings=np.vstack(embeddings))
    print(f"Saved embeddings to {embedding_output_path}")