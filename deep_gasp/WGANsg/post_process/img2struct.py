import os
import csv
import ast
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from pymatgen.core import Lattice, Structure, Molecule, Element
from pymatgen.io.vasp.outputs import Poscar
from pymatgen.transformations.standard_transformations import RotationTransformation

from ase import Atoms
from ase.io import read, write
from ase.build import molecule


class POSCAR():
    def __init__(self, images, composition, path):
        # Pairwise distance matrices and max dimension
        self.images = images 
        self.elem_list = composition
        self.path = path

    def extract_dims(self, images, elem_list):
        """
        Extracts the number of atomic species from the first line of each image 
        using a threshold based on the minimum atomic number in elem_list.

        Parameters:
        -----------
        images : list of np.ndarray
            List of image arrays representing encoded structure information.
        elem_list : list of str
            List of element symbols used in the structures.

        Returns:
        --------
        dims : list of int
            List of inferred atomic counts per image based on thresholding.
        """
        atomic_numbers = [Element(el).Z for el in elem_list]
        threshold = min(atomic_numbers) / 4

        dims = []
        for image in images:
            first_line = image[0]
            count = sum(1 for pixel in first_line if pixel > threshold)
            dims.append(count)

        return dims

    def extract_atomic_numbers(self, generated_structures, dims):
        """
        Extracts atomic number encodings from the first line of each structure image,
        truncated to the predicted number of atoms.

        Parameters:
        -----------
        generated_structures : list of np.ndarray
            List of image arrays representing generated structures.
        dims : list of int
            Number of atoms per structure.

        Returns:
        --------
        extracted_atomic_numbers : list of np.ndarray
            Extracted atomic number arrays per structure.
        """
        extracted_atomic_numbers = []
        for count, generated_structure in enumerate(generated_structures):
            extracted_line = generated_structure[0][:dims[count]]
            extracted_atomic_numbers.append(extracted_line)
        return extracted_atomic_numbers

    def correct_species(self, extracted_atomic_numbers, elem_list):
        """
        Maps predicted atomic number encodings to the closest real atomic numbers 
        from elem_list and returns the corresponding element symbols.

        Parameters:
        -----------
        extracted_atomic_numbers : list of np.ndarray
            Atomic number predictions per structure.
        elem_list : list of str
            Allowed chemical elements.

        Returns:
        --------
        corrected_species_symbols : list of list of str
            Corrected atomic species as element symbols for each structure.
        """
        species_list = [[round(float(an)) for an in atomic_numbers]
                        for atomic_numbers in extracted_atomic_numbers]
        reference_species = [Element(element).number for element in elem_list]
        
        corrected_species_list = [self.closest_number(reference_species, species)
                                  for species in species_list]

        corrected_species_symbols = [[Element.from_Z(atomic_number).name
                                      for atomic_number in species]
                                     for species in corrected_species_list]

        return corrected_species_symbols

    def closest_number(self, reference, numbers):
        """
        For each number in `numbers`, finds the closest number in the `reference` list.

        Parameters:
        -----------
        reference : list of int
            List of valid atomic numbers.
        numbers : list of int
            Predicted atomic numbers to be corrected.

        Returns:
        --------
        min_vals : list of int
            List of corrected atomic numbers based on closest match.
        """
        min_vals = []
        for i in numbers:
            min_val = min(reference, key=lambda x: abs(x - i))
            min_vals.append(min_val)
        return min_vals

    def get_coordinates(self, images, dims):
        """
        Extracts and reshapes atomic coordinates from image representations.

        Parameters:
        -----------
        images : list of np.ndarray
            List of image arrays containing atomic coordinate information.
        dims : list of int
            Number of atoms per image.

        Returns:
        --------
        truncated_coords : list of np.ndarray
            List of reshaped atomic coordinate arrays.
        """
        coordinates = [image[1:4] for image in images]
        truncated_coords = [
            np.array([coord[:dims[i]] for coord in extracted_coordinate]).reshape(-1, 3)
            for i, extracted_coordinate in enumerate(coordinates)
        ]
        return truncated_coords

    def get_lattice_parameters(self, images):
        """
        Extracts average lattice constants and angles from encoded image arrays.

        Parameters:
        -----------
        images : list of np.ndarray
            List of image arrays where indices 4–9 contain lattice information.

        Returns:
        --------
        lattice_parameters : list of list of float
            Each entry is [a, b, c, alpha, beta, gamma] for one structure.
        """
        a = [np.mean(image[4]) for image in images]
        b = [np.mean(image[5]) for image in images]
        c = [np.mean(image[6]) for image in images]
        alpha = [np.mean(image[7]) for image in images]
        beta = [np.mean(image[8]) for image in images]
        gamma = [np.mean(image[9]) for image in images]
        lattice_parameters = [[a[i], b[i], c[i], alpha[i], beta[i], gamma[i]]
                              for i in range(len(images))]
        return lattice_parameters

    def create_poscar(self, coords, species, parameters, path):
        """
        Constructs Structure objects using atomic coordinates, species, and lattice
        parameters, and writes them to POSCAR files.

        Parameters:
        -----------
        coords : list of np.ndarray
            List of atomic coordinates for each structure.
        species : list of list of str
            Atomic species for each structure.
        parameters : list of list of float
            Lattice parameters [a, b, c, alpha, beta, gamma] for each structure.
        path : str
            Directory path to save the generated POSCAR files.

        Returns:
        --------
        structure_list : list of pymatgen.Structure
            List of generated Structure objects.
        """
        lattice_parameters = [Lattice.from_parameters(param[0], param[1], param[2],
                                                      param[3], param[4], param[5])
                              for param in parameters]
        structure_list = []
        count = 0 
        for i in range(len(coords)):
            try:
                structure = Structure(lattice_parameters[i], species[i], coords[i])
                structure_list.append(structure)
                poscar = Poscar(structure)
                destination = path + f'POSCAR_{count}'
                poscar.write_file(destination)
                count = count + 1
            except:
                pass
        structure_list = [Structure(lattice_parameters[i], species[i], coords[i])
                          for i in range(len(coords))]

        return structure_list
    
    def convert_to_poscars(self):
        try:
            os.mkdir(self.path)
        except:
            print(f'-- {self.path} already exists')
        dims = self.extract_dims(self.images,self.elem_list)
        atomic_numbers = self.extract_atomic_numbers(self.images,dims)
        species = self.correct_species(atomic_numbers,self.elem_list)
        coords = self.get_coordinates(self.images,dims)
        lattice_parameters = self.get_lattice_parameters(self.images)
        structures = self.create_poscar(coords,species,lattice_parameters,self.path)
        print(f'-- sucessfully converted {len(self.images)} images into structures within {self.elem_list} composition in {self.path}!')
        return structures
