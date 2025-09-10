10-09-2025

I will finish this soon! I am very tired and have been at this for a while! Needs updating for usage, instructions and requirements.

# halluciTATE

This project is a refactored pipeline of an initial MSc project, built to be reproducible and scalable for input of large numbers of proteins. The aim of the project is to be able to identify AI generated proteins that are hallucinated. It is also a pun on my surname.

Work is currently in progress to refactor existing code. Bear with me.

# Requires (definitely not finished!):

- BioPython
- Hugging Face
- EvoProtGrad
- Phenix Molprobity
- Python 3.9
- Python 3.12

# EvoProtGrad

I would recommend using a high performance cluster to use EvoProtGrad this way with the larger expert models. I used a Tesla A100 and it could take a couple of days to get 10 proteins in a family to run until convergence.


# Usage 

# Downloading fastas
After downloading your desired PDB IDs from interprot in TSV format, run

python -m pipeline.download_fastas

# Embed sequences

# Cluster sequences
Example usage below
python pipeline.py --protein_family kinase --clusters 10 --output results/kinase
