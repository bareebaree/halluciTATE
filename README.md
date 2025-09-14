Report is currently not published as it is under review, but the code for replicating the analysis is contained in this repo.

# halluciTATE

This project is a refactored pipeline of an initial MSc project, built to be reproducible and scalable for input of large numbers of proteins. The aim of the project is to be able to identify AI generated proteins that are hallucinated. It is also a pun on my surname.

Work is currently in progress to refactor existing code. Bear with me.

# Requires:

- BioPython
- Hugging Face
- EvoProtGrad
- Phenix Molprobity (clashscore)
- Python 3.9
- Python 3.12
- temBERTure (requires separate environment with python 3.9)
  
Follow instructions for Phenix Molprobity, Hugging Face, EvoProtGrad, and temBERTure from their respective publishers.


# EvoProtGrad

I would recommend using a high performance cluster to use EvoProtGrad this way with the larger expert models. I used a Tesla A100 and it could take a couple of days to get 10 proteins in a family to run until convergence.


# Usage 

Run from a root directory. ./pipeline, ./analysis expect ./data, ./results, and ./supplementary_data to be in ./root (or whatever you call it) 

# 1. Downloading fastas
After downloading your desired PDB IDs from interprot in TSV format, run

python -m pipeline.download_fastas

# 2. Embed sequences
Example usage below

python -m pipeline.embed --family kinase

# 3. Cluster sequences
Example usage below

python -m pipeline --family kinase --clusters 10

# 4. Evolve

python -m pipeline --family kinase --stage evolve

# 5. Select best score from each iteration of evolve loop

python -m pipeline --family kinase --stage summarise

# 6. Convert fasta to CSV for analysis
python pipeline/fasta_to_csv.py

# 7. Run temBERTure

python -m pipeline --family kinase --stage temberture


# 8 Analysis and visualisation

Going through the analyse_sequences.ipynb notebook will reproduce the analysis in the paper

Going through the analyse_structures.ipynb after analyse_sequencess will reproduce the analysis on structures in the paper. Foldseek should be installed independently.

Replace the hard-coded filepaths in with where many of these additional programmes have stored data and results in these notebooks. In future builds I will build relative paths in to handle these.





