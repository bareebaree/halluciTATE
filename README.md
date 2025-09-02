# halluciTATE

This project is a refactored pipeline of an initial MSc project, built to be reproducible and scalable for input of large numbers of proteins. The aim of the project is to be able to identify AI generated proteins that are hallucinated. It is also a pun on my surname.



# Requires (definitely not finished!):

- BioPython
- Hugging Face
- EvoProtGrad
- Phenix Molprobity

# EvoProtGrad

I would recommend using a high performance cluster to use EvoProtGrad this way with the larger expert models. I used a Tesla A100 and it could take a couple of days to get 10 proteins in a family to run until convergence.
