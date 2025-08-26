# DEEP-GASP
This repository contains the code of the **DEEP-GASP** algorithm, an end-to-end machine learning integrated genetic algorithm for crystal structure prediction, containing a Wasserstein Generative Adversarial Network Structure Generator (WGANsg) 
for population initialization and tournament based scoring/selection (TBS) of candidate offspring organisms through equivariant graph based score networks via MatterSim. 


<img width="2500" height="642" alt="smart_gasp_github" src="https://github.com/user-attachments/assets/69dec0a7-e851-4cd6-9af8-b613b070a4b5" />


# Getting DEEP-GASP

## 1. Create Conda Environment
- While not mandatory, we recommend creating a clean conda environment before installing DEEP-GASP to avoid potential package conflicts. You can create and activate a conda environment with the following commands:
- ```bash conda create -n smart_gasp python=3.10 -y```
- 
### Activate the environment
- ```conda activate smart_gasp```

## 2. Install required packages
- ```pip install -r requirements.txt```

## 3. Install from source
- ```git clone https://github.com/samdong2101/DEEP_GASP.git```
