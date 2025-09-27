# DEEP-GASP
This repository contains the code of the **DEEP-GASP** algorithm, an end-to-end machine learning integrated genetic algorithm for crystal structure prediction, containing a Wasserstein Generative Adversarial Network Structure Generator (WGANsg) 
for population initia<img width="793" height="366" alt="deep_gasp_workflow" src="https://github.com/user-attachments/assets/da3bafa5-713c-4b7e-858c-b6ddee33fa5d" />
lization and tournament based scoring/selection (TBS) of candidate offspring organisms through equivariant graph based score networks via MatterSim. 

<img width="2500" height="1142" alt="deep_gasp_workflow_github" src="https://github.com/user-attachments/assets/22473d53-d62e-4971-883d-bcea33a58a9d" />



# Getting DEEP-GASP

## 1. Create Conda Environment
- While not mandatory, we recommend creating a clean conda environment before installing DEEP-GASP to avoid potential package conflicts. You can create and activate a conda environment with the following commands:
- ```conda create -n deep_gasp python=3.10```
### Activate the environment
- ```conda activate deep_gasp```

## 2. Install required packages
- ```pip install -r requirements.txt```

## 3. Install from source
- ```git clone https://github.com/samdong2101/DEEP_GASP.git```
