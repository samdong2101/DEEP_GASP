# Core Python
import os
import io
import csv
import ast
import random
import fnmatch
import pickle

# Numerical and Data Handling
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt  # Redundant — remove one (see note below)

# ASE (Atomic Simulation Environment)
from ase import Atoms
from ase.io import read, write
from ase.build import molecule

# Pymatgen
from pymatgen.core import Lattice, Structure, Molecule, Element
from pymatgen.transformations.standard_transformations import RotationTransformation

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch

class PreprocessData:

    def __init__(self,structures_list,composition,max_size,min_size,augment = True, min_energy = 0.1):
        # Pairwise distance matrices and max dimension
        self.structures_list = structures_list
        self.elem_list = composition
        self.max_size = max_size
        self.min_size = min_size
        self.augment = augment
        self.min_energy = min_energy
        assert self.min_size < self.max_size, 'min_size is greater than max_size'
    def extract_structures(self,structures_list, elem_list):
        """
        Filters a list of structures, retaining only those composed entirely of the specified elements.

        Parameters:
        -----------
        structures_list : list
            A list of Pymatgen Structure objects to be filtered.
        elem_list : list of str
            A list of allowed element symbols (e.g., ['Nb', 'O', 'Pt']).
            Only structures containing atoms exclusively from this list will be retained.

        Returns:
        --------
        extracted_structures : list
            A list of Structure objects where every atomic site is one of the specified elements.
        """
        allowed_atomic_numbers = {Element(e).Z for e in elem_list}

        extracted_structures = []
        for structure in structures_list:
            structure_atomic_numbers = set(structure.atomic_numbers)
            if structure_atomic_numbers.issubset(allowed_atomic_numbers):
                extracted_structures.append(structure)

        return extracted_structures

    def filter_structures(self,dataset,max_size,min_size,min_energy):
        """
        Filters a list of structures based on their number of atoms.

        Parameters:
        -----------
        dataset : list
            A list of structures (e.g., Pymatgen Structure objects).
        max_size : int
            The maximum allowed number of atoms in a structure.
        min_size : int
            The minimum allowed number of atoms in a structure.

        Returns:
        --------
        filtered_structures : list
            A list of structures whose number of atoms is strictly greater than min_size
            and less than or equal to max_size.
        """
        filtered_energy_dataset = [i[0] for i in dataset if i[1] is not None and i[1]<=self.min_energy]
        filtered_structures = []
        for i in filtered_energy_dataset:
            x = i.distance_matrix.shape[0]
            if x<=max_size and x> min_size:
                filtered_structures.append(i)
        return filtered_structures

    def generate_rotated_structures(self,structures, num_angles=15):
        """
        Applies 3D rotations to a list of structures around the Cartesian axes.

        Parameters:
        -----------
        structures : list
            A list of Pymatgen Structure objects to rotate.
        num_angles : int, optional (default=15)
            The number of evenly spaced rotation angles (in degrees) between 0 and 360
            to apply around each axis.

        Returns:
        --------
        rotated_data_set : list
            A list of rotated Structure objects. For each input structure, the function returns
            `3 * num_angles` rotated versions (3 axes × num_angles per axis).
        """
        axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        angles = np.linspace(0, 360, num_angles).tolist()

        rotated_data_set = []

        for structure in structures:
            for axis in axes:
                for angle in angles:
                    rotation = RotationTransformation(axis, angle)
                    rotated = rotation.apply_transformation(structure)
                    rotated_data_set.append(rotated)
        return rotated_data_set

    def preprocess_data(self):
        print(f'-- Filtering {len(self.structures_list)} structures < {self.min_energy} eV with max {self.max_size} atoms and min {self.min_size} atoms...')
        filtered_structures = self.filter_structures(self.structures_list, self.max_size, self.min_size, self.min_energy)
        print(f'-- Extracting structures with composition {self.elem_list}...')
        extracted_structures = self.extract_structures(filtered_structures,self.elem_list)
        if self.augment:
            print(f'-- Augmenting...')
            rotated_data_set = self.generate_rotated_structures(extracted_structures)
            print('-- Done preprocessing! ')
            return rotated_data_set
        else:
            print('-- Done preprocessing! ')
            return filtered_structures


class PNGrepresentation:
    def __init__(self,structures_list,bool_frac_coords = True):
        # Pairwise distance matrices and max dimension
        self.pwds = None        # List of padded pairwise distance matrices
        self.max_dim = None     # Max number of atoms among all structures
        self.structures_list = structures_list
        # Atomic numbers
        self.atomic_numbers = None   # 2D array of padded atomic numbers

        # Fractional coordinates
        self.x_coords = None
        self.y_coords = None
        self.z_coords = None

        # Lattice parameters (a, b, c)
        self.a_parameters = None
        self.b_parameters = None
        self.c_parameters = None

        # Lattice angles (alpha, beta, gamma)
        self.alphas = None
        self.betas = None
        self.gammas = None

        # Cell volumes
        self.volumes = None
        self.pngs = None
        self.png_dim1 = None
        self.png_dim2 = None
        # Extraneous
        self.bool_frac_coords = bool_frac_coords
        self.divisor_list = []
        self.factor_list = []
        # Any other attributes can be added here as needed
    def get_pairwise_distances(self,structures):

        """
        Computes and pads pairwise distance matrices for a list of structures.

        Parameters:
        -----------
        structures : list
            A list of Pymatgen Structure objects.

        Returns:
        --------
        padded_distance_mats : list of np.ndarray
            A list of 2D NumPy arrays, each representing a padded pairwise distance matrix.
            Each matrix has shape (max_dim, max_dim), where `max_dim` is the size of the largest structure.
            All distances are scaled by a factor of 15.

        max_dim : int
            The maximum number of atomic sites among all structures (i.e., the largest original dimension
            of any distance matrix). This is also the shape of each padded matrix.
        """


        distance_mats = [structure.distance_matrix*15 for structure in structures]
        dim_mat = [structure.distance_matrix.shape[0] for structure in structures]
        max_dim = max(dim_mat)
        pwds_mats = []
        for dis_mat in distance_mats:
            rows_padding = max_dim-dis_mat.shape[0]
            cols_padding = max_dim-dis_mat.shape[1]
            padded_distance_mat = np.pad(dis_mat,((0,rows_padding),(0,cols_padding)),mode = 'constant')
            pwds_mats.append(padded_distance_mat)
        self.pwds = pwds_mats
        self.max_dim = max_dim
        return pwds_mats,max_dim

    def get_atomic_numbers(self,structures,max_dim):

        """
        Extracts and pads atomic number arrays from a list of structures.

        Parameters:
        -----------
        structures : list
            A list of Pymatgen Structure objects.
        max_dim : int
            The target length for each atomic number array. Structures with fewer atoms will be
            zero-padded to this length.

        Returns:
        --------
        atomic_numbers : np.ndarray
            A 2D NumPy array of shape (num_structures, max_dim), where each row contains the
            atomic numbers for a structure, padded with zeros as needed. The values are scaled
            by a factor of 4.
        """

        atomic_numbers = []
        for structure in structures:
            atomic_numbers_arr = np.array(structure.atomic_numbers)
            padded_atomic_numbers = np.concatenate([
                atomic_numbers_arr,
                np.zeros(max_dim - len(atomic_numbers_arr), dtype=atomic_numbers_arr.dtype)
            ])
            atomic_numbers.append(padded_atomic_numbers)
        self.atomic_numbers = np.array(atomic_numbers)*4
        self.factor_list.append(4)
        return atomic_numbers

    def add_coordinates(self,structure_list,max_dim,frac_coords = True):

        """
        Extracts and pads fractional atomic coordinates from a list of structures.

        Parameters:
        -----------
        structure_list : list
            A list of Pymatgen Structure objects from which fractional coordinates
            will be extracted.
        max_dim : int
            The target length for padding each coordinate array. Structures with fewer
            atoms will have their coordinate lists zero-padded to this length.

        Returns:
        --------
        x_frac, y_frac, z_frac : lists of lists
            Three lists containing the padded fractional coordinates along the x, y, and z axes,
            respectively. Each inner list corresponds to a structure’s coordinates, scaled and
            padded to length `max_dim`.
        """
        if frac_coords:
            coords_list = [structure.frac_coords*125 for structure in structure_list]
            self.factor_list.append(125)
            self.factor_list.append(125)
            self.factor_list.append(125)
        else:
            coords_list = [structure.frac_coords*25 for structure in structure_list]
            self.factor_list.append(25)
            self.factor_list.append(25)
            self.factor_list.append(25)
        x_coords = [list(np.pad([c[0] for c in coord], (0, max_dim - len(coord)))) for coord in coords_list]
        y_coords = [list(np.pad([c[1] for c in coord], (0, max_dim - len(coord)))) for coord in coords_list]
        z_coords = [list(np.pad([c[2] for c in coord], (0, max_dim - len(coord)))) for coord in coords_list]
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.z_coords = z_coords
        return x_coords,y_coords,z_coords

    def add_lattice_constants(self, structure_list, max_dim, frac_coords):
        """
        Extracts and pads lattice parameters (a, b, c) from a list of structures.

        Parameters:
        -----------
        structure_list : list
            A list of Pymatgen Structure objects from which lattice parameters will be extracted.
        max_dim : int
            The target length for padding each lattice parameter array. Each parameter value
            will be repeated and padded to this length.

        Returns:
        --------
        a_parameters, b_parameters, c_parameters : lists of np.ndarray
            Three lists containing padded lattice parameters 'a', 'b', and 'c', respectively.
            Each entry corresponds to a structure, where the scalar lattice parameter is
            expanded into an array of length `max_dim`. The padding is effectively zero, but
            since zeros are multiplied by 10, the padding remains zero.
        """
        a_parameters = [(structure.lattice.abc[0]+np.zeros(max_dim))*10 for structure in structure_list]
        b_parameters = [(structure.lattice.abc[1]+np.zeros(max_dim))*10 for structure in structure_list]
        c_parameters = [(structure.lattice.abc[2]+np.zeros(max_dim))*10 for structure in structure_list]
        self.a_parameters = a_parameters
        self.b_parameters = b_parameters
        self.c_parameters = c_parameters
        self.factor_list.append(10)
        self.factor_list.append(10)
        self.factor_list.append(10)
        return a_parameters,b_parameters,c_parameters

    def add_lattice_angles(self,structure_list,max_dim):

        """
        Extracts and pads lattice angles (alpha, beta, gamma) from a list of structures.

        Parameters:
        -----------
        structure_list : list
            A list of Pymatgen Structure objects from which lattice angles will be extracted.
        max_dim : int
            The target length for padding each lattice angle array. Each angle value will be
            repeated and padded to this length.

        Returns:
        --------
        alpha, beta, gamma : lists of np.ndarray
            Three lists containing padded lattice angles alpha, beta, and gamma, respectively.
            Each entry corresponds to a structure, where the scalar lattice angle is expanded
            into an array of length `max_dim`.

        """

        alphas = [structure.lattice.angles[0]+np.zeros(max_dim) for structure in structure_list]
        betas = [structure.lattice.angles[1]+np.zeros(max_dim) for structure in structure_list]
        gammas = [structure.lattice.angles[2]+np.zeros(max_dim) for structure in structure_list]
        self.alphas = alphas
        self.betas = betas
        self.gammas = gammas
        self.factor_list.append(1)
        self.factor_list.append(1)
        self.factor_list.append(1)
        return alphas, betas, gammas
    def add_cell_volume(self,structure_list,max_dim):
        """
        Extracts and pads the cell volume from a list of structures.

        Parameters:
        -----------
        structure_list : list
            A list of Pymatgen Structure objects from which cell volumes will be extracted.
        max_dim : int
            The target length for padding each volume array. Each scalar volume value will
            be expanded into an array of this length.

        Returns:
        --------
        padded_volumes : list of np.ndarray
            A list where each element is an array of length `max_dim` filled with the cell
            volume (scaled by 1/5) of the corresponding structure.
        """

        volumes = [np.zeros(max_dim) + structure.volume/5 for structure in structure_list]
        self.volumes = volumes
        self.factor_list.append(0.2)
        return volumes

    def compile_pngs(self):
        pngs = [np.vstack([self.atomic_numbers[i],self.x_coords[i],self.y_coords[i],self.z_coords[i],
                            self.a_parameters[i],self.b_parameters[i],
                           self.c_parameters[i], self.alphas[i], self.betas[i], self.gammas[i],
                           self.volumes[i],self.pwds[i]]) for i in range(len(self.structures_list))]
        self.pngs = torch.tensor(pngs)
        pngs = torch.tensor(pngs)
        return pngs

    def truncate_pngs(self):
        truncated_pngs = self.pngs[:, 0:11, :]
        self.pngs = truncated_pngs
        return truncated_pngs

    def normalize_pngs(self):
        # self.pngs assumed shape: (n, dim1, dim2)

        # Compute max per row across all PNGs (axis 0 = images, axis 2 = columns)
        # Resulting shape: (dim1,)
        self.divisor_list = self.pngs.max(dim=0).values.max(dim=1).values
        self.pngs = self.pngs / self.divisor_list.view(1, -1, 1)



    def featurize(self):
        print(f'-- Featurizing {len(self.structures_list)} structures into images...')
        pwds,max_dim = self.get_pairwise_distances(self.structures_list)
        atomic_numbers = self.get_atomic_numbers(self.structures_list,max_dim)
        x,y,z = self.add_coordinates(self.structures_list,max_dim)
        a,b,c = self.add_lattice_constants(self.structures_list,max_dim,self.bool_frac_coords)
        alpha,beta,gamma = self.add_lattice_angles(self.structures_list,max_dim)
        volume_png = self.add_cell_volume(self.structures_list,max_dim)
        pngs = self.compile_pngs()
        truncated_pngs = self.truncate_pngs()
        self.normalize_pngs()
        print(f'-- Featurization complete!')
        self.png_dim1, self.png_dim2 = self.pngs.shape[1], self.pngs.shape[2]
        return self.pngs,self.png_dim1,self.png_dim2,self.divisor_list,self.factor_list

