# WGANs-Integrated-Genetic-Algorithms
This repository contains the code of the **W**asserstein **G**enerative **A**dversarial **S**tructure **G**enerator (WGAN-sg) designed to create stronger starting initial populations for genetic algorithms for crystal structure and phase prediction.

<img width="2500" height="1045" alt="github_workflow" src="https://github.com/user-attachments/assets/399db30c-07c5-4c6e-88a6-06eee4475b90" />

The WGAN-sg is trained on thermodynamically stable and metastable crystals (<0.1 eV/atom), where crystals are represented in a png image format. This png representation takes crystal parameters such as element type, fractional coordinates, lattice parameters etc. and encodes these values into the pixels of an image, where pixel brightness indicates magnitude.

<img width="1555" height="1224" alt="github_wgans_architecture" src="https://github.com/user-attachments/assets/b8a07955-4774-4509-b3b5-a8243e7e3aac" />


- Data preparation and model training can be executed within run.py, which can be executed via

```python train.py --config /path/to/input_file``` 

- Similarly, if you would like to generate materials from a pre-trained model, this can be executed via

```python generate.py --config /path/to/input_file```

- The input file for run.py is a json containing a dictionary of parameters, an example of the input file can be found in this repository under

```WGANsg/input.json```
- The contents within the input file are as follows
    - ```structures_path: (str) a path to an array-type of pymatgen structures```
    - ```elem_list: (list) A list of strings containing atomic symbols of your desired composition space i.e. ['Cu','Al']```
    - ```max_atoms: (int) the maximum number of atoms N you want in your data```
    - ```min_atoms: (int) the minimum number of atoms N you want in your data```
    - ```g_lr: (float) the learning rate of the generator network```
    - ```c_lr: (float) the learning rate of the critic network```
    - ```epochs_inner: (int) the number of epochs you would like for training before each evaluation loop```
    - ```epochs_outer: (int) the total number of evaluation loops you would like for training```
    - ```batch_size: (int) batch size of data```
    - ```generator_weights_path: (str) the desired destination to save the weights of your generator network```
    - ```num_images: (int) the total number of images (crystal structures) you want to generate```
    - ```poscar_path: (str) the desired destination of your converted crystal structures made by the WGAN-sg```
