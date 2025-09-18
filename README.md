# DEEP-GASP
This repository contains the code of the **DEEP-GASP** algorithm, an end-to-end machine learning integrated genetic algorithm for crystal structure prediction, containing a Wasserstein Generative Adversarial Network Structure Generator (WGANsg) 
for population initialization and tournament based scoring/selection (TBS) of candidate offspring organisms through equivariant graph based score networks via MatterSim. 

<img width="2412" height="1109" alt="deep_gasp_workflow_github" src="https://github.com/user-attachments/assets/fae15470-2be4-454f-b9b8-13994db3cfd2" />



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
