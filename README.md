
# halluciTATE
This project sets out to identify features of hallucinations in AI generated protein sequences, and by extension, structures.
The first step, identifying features of hallucinations has been finished. The next step, is to identify knowledge neurons that lead to decision making for making models hallucinate.

The below image demonstrates where in a sequence hallucinations are likely in purposefully hallucinated, overfitted sequences. Blue demonstrates a region with lower aln-TM score on Foldseek, orange, higher. Green lines demonstrate homorepeat motifs, which are a common feature in hallucinations.

<img width="582" height="343" alt="image" src="https://github.com/user-attachments/assets/f777a655-2a4c-4614-ad67-5c2c3e94aa33" />


This project is a refactored pipeline of an initial MSc project, built to be reproducible and scalable for input of large numbers of proteins. The aim of the project is to be able to identify AI generated proteins that are hallucinated. It is also a pun on my surname.


# Requires:

Requirements are in the .yml files in the repo. For almost all steps, the general_environment.yml should be used. For the temBERTure step, the temberture_environment.yml should be used.
  
Follow instructions for Phenix Molprobity, Hugging Face, EvoProtGrad, and temBERTure from their respective publishers.


# EvoProtGrad

I would recommend using a high performance cluster to use EvoProtGrad this way with the larger expert models. I used a Tesla A100 and it could take a couple of days to get 10 proteins in a family to run until convergence.


# Usage 

Run from a root directory. ./pipeline, ./analysis expect ./data, ./results, and ./supplementary_data to be in ./root (or whatever you call it) 

# 1. Downloading fastas
After downloading your desired PDB IDs from interprot in TSV format, run

python -m pipeline.download_fastas --tsv <tsv_file>

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

Unzip the foldseek results .zip file for this step.

Going through the analyse_sequences.ipynb notebook will reproduce the analysis in the paper

Going through the analyse_structures.ipynb after analyse_sequencess will reproduce the analysis on structures in the paper. Foldseek should be installed independently.

Replace the hard-coded filepaths in with where many of these additional programmes have stored data and results in these notebooks. In future builds I will build relative paths in to handle these.





