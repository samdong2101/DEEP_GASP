
from __future__ import division, unicode_literals, print_function


"""
Geometry module:

This module contains classes to hold geometry-specific data and operations,
including any additional constraints. All geometry classes must implement
pad(), unpad() and get_size() methods.

1. Bulk: Data and operations for 3D bulk structures

2. Sheet: Data and operations for 2D sheet structures

3. Wire: Data and operations for 1D wire structures

4. Cluster: Data and operations for 0D cluster structures

"""

from pymatgen.core.lattice import Lattice
from pymatgen.core.sites import Site

import numpy as np
import copy


class Bulk(object):
    '''
    Contains data and operations specific to bulk structures (so not much...).
    '''

    def __init__(self):
        '''
        Makes a Bulk object.
        '''

        self.shape = 'bulk'
        self.max_size = np.inf
        self.min_size = -np.inf
        self.padding = None

    def pad(self, cell, padding='from_geometry'):
        '''
        Does nothing.

        Args:
            cell: the Cell to pad

            padding: the amount of vacuum padding to add. If set to
                'from_geometry', then the value in self.padding is used.
        '''

        pass

    def unpad(self, cell, n_sub, constraints):
        '''
        Does nothing.

        Args:
            cell: the Cell to unpad

            constraints: the Constraints of the search
        '''

        pass

    def get_size(self, cell):
        '''
        Returns 0.

        Args:
            cell: the Cell whose size to get
        '''

        return 0
class Sheet(object):
    '''
    Contains data and operations specific to sheet structures.
    '''

    def __init__(self, geometry_parameters):
        '''
        Makes a Sheet, and sets default parameter values if necessary.

        Args:
            geometry_parameters: a dictionary of parameters
        '''

        self.shape = 'sheet'

        # default values
        self.default_max_size = np.inf
        self.default_min_size = -np.inf
        self.default_padding = 10

        # parse the parameters, and set defaults if necessary
        # max size
        if 'max_size' not in geometry_parameters:
            self.max_size = self.default_max_size
        elif geometry_parameters['max_size'] in (None, 'default'):
            self.max_size = self.default_max_size
        else:
            self.max_size = geometry_parameters['max_size']

        # min size
        if 'min_size' not in geometry_parameters:
            self.min_size = self.default_min_size
        elif geometry_parameters['min_size'] in (None, 'default'):
            self.min_size = self.default_min_size
        else:
            self.min_size = geometry_parameters['min_size']

        # padding
        if 'padding' not in geometry_parameters:
            self.padding = self.default_padding
        elif geometry_parameters['padding'] in (None, 'default'):
            self.padding = self.default_padding
        else:
            self.padding = geometry_parameters['padding']

    def pad(self, cell, padding='from_geometry'):
        '''
        Modifies a cell by adding vertical vacuum padding and making the
        c-lattice vector normal to the plane of the sheet. The atoms are
        shifted to the center of the padded sheet.

        Args:
            cell: the Cell to pad

            padding: the amount of vacuum padding to add (in Angstroms). If not
                set, then the value in self.padding is used.
        '''

        # get the padding amount
        if padding == 'from_geometry':
            pad_amount = self.padding
        else:
            pad_amount = padding

        # make the padded lattice
        cell.rotate_to_principal_directions()
        species = cell.species
        cartesian_coords = cell.cart_coords
        cart_bounds = cell.get_bounding_box(cart_coords=True)
        minz = cart_bounds[2][0]
        maxz = cart_bounds[2][1]
        layer_thickness = maxz - minz
        ax = cell.lattice.matrix[0][0]
        bx = cell.lattice.matrix[1][0]
        by = cell.lattice.matrix[1][1]
        padded_lattice = Lattice([[ax, 0.0, 0.0], [bx, by, 0.0],
                                  [0.0, 0.0, layer_thickness + pad_amount]])

        # modify the cell to correspond to the padded lattice
        cell.lattice = padded_lattice
        site_indices = []
        for i in range(len(cell.sites)):
            site_indices.append(i)
        cell.remove_sites(site_indices)
        for i in range(len(cartesian_coords)):
            cell.append(species[i], cartesian_coords[i],
                        coords_are_cartesian=True)

        # translate the atoms back into the cell if needed, and shift them to
        # the vertical center
        cell.translate_atoms_into_cell()
        frac_bounds = cell.get_bounding_box(cart_coords=False)
        z_center = frac_bounds[2][0] + (frac_bounds[2][1] -
                                        frac_bounds[2][0])/2
        translation_vector = [0, 0, 0.5 - z_center]
        site_indices = [i for i in range(len(cell.sites))]
        cell.translate_sites(site_indices, translation_vector,
                             frac_coords=True, to_unit_cell=False)

    def unpad(self, cell, n_sub, constraints):
        '''
        Modifies a cell by removing vertical vacuum padding, leaving only
        enough to satisfy the per-species MID constraints, and makes the
        c-lattice vector normal to the plane of the sheet (if it isn't
        already).

        Args:
            cell: the Cell to unpad

            constraints: the Constraints of the search
        '''

        # make the unpadded lattice
        cell.rotate_to_principal_directions()
        species = cell.species
        cartesian_coords = cell.cart_coords
        layer_thickness = self.get_size(cell)
        max_mid = constraints.get_max_mid() + 0.01  # just to be safe...
        ax = cell.lattice.matrix[0][0]
        bx = cell.lattice.matrix[1][0]
        by = cell.lattice.matrix[1][1]
        unpadded_lattice = Lattice([[ax, 0.0, 0.0], [bx, by, 0.0],
                                    [0.0, 0.0, layer_thickness + max_mid]])

        # modify the cell to correspond to the unpadded lattice
        cell.lattice = unpadded_lattice
        site_indices = []
        for i in range(len(cell.sites)):
            site_indices.append(i)
        cell.remove_sites(site_indices)
        for i in range(len(cartesian_coords)):
            cell.append(species[i], cartesian_coords[i],
                        coords_are_cartesian=True)

        # translate the atoms back into the cell if needed, and shift them to
        # the vertical center
        cell.translate_atoms_into_cell()
        frac_bounds = cell.get_bounding_box(cart_coords=False)
        z_center = frac_bounds[2][0] + (frac_bounds[2][1] -
                                        frac_bounds[2][0])/2
        translation_vector = [0, 0, 0.5 - z_center]
        site_indices = [i for i in range(len(cell.sites))]
        cell.translate_sites(site_indices, translation_vector,
                             frac_coords=True, to_unit_cell=False)

    def get_size(self, cell):
        '''
        Returns the layer thickness of a sheet structure, which is the maximum
        vertical distance between atoms in the cell.

        Precondition: the cell has already been put into sheet format (c
            lattice vector parallel to the z-axis and a and b lattice vectors
            in the x-y plane)

        Args:
            cell: the Cell whose size to get
        '''

        cart_bounds = cell.get_bounding_box(cart_coords=True)
        layer_thickness = cart_bounds[2][1] - cart_bounds[2][0]
        return layer_thickness

