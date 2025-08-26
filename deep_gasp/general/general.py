

# coding: utf-8
# Copyright(c) Henniggroup.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals, print_function

"""
General module:

This module contains several classes central to the algorithm.

1. IDGenerator: generates consecutive integer ID numbers for organisms

2. Organism: represents an organism, which consists primarily of a
        structure and an energy

3. Cell: represents a structure, and extends pymatgen.core.structure.Structure

4. OffspringGenerator: high-level class for making offspring organisms

5. SelectionProbDist: specifies the distribution from which selection
        probabilities are drawn

6. CompositionSpace: specifies the composition or composition range

7. CompositionFitnessWeight: specifies how the weight given to the composition
        fitness is to be computed

8. StoppingCriteria: specifies when a search should stop

9. DataWriter: writes information about the search to a file

"""

from pymatgen.core.structure import Structure, Molecule
from pymatgen.core.lattice import Lattice
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element, DummySpecie
from pymatgen.analysis.phase_diagram import CompoundPhaseDiagram
from pymatgen.analysis.phase_diagram import PDEntry
from pymatgen.transformations.standard_transformations import \
    RotationTransformation

import numpy as np
from collections import defaultdict
import random
class IDGenerator(object):
    """
    Generates successive integer ID numbers, starting from 1.
    """

    def __init__(self):
        """
        Makes an IDGenerator.
        """

        self.id = 0

    def make_id(self):
        """
        Returns the next id number.
        """

        self.id += 1
        return self.id


class Organism(object):
    """
    An organism, consisting primarily of a cell and an energy, as well as
    several derived quantities.
    """

    def __init__(self, cell, id_generator, maker, composition_space):
        """
        Makes an Organism.

        Args:
            cell: the cell of this organism, as a Cell object

            id_generator: the IDGenerator used to assign id numbers to all
                organisms

            maker: the name of algorithm that made the organism, as a string.
                Either a creator or a variation

            composition_space: the CompositionSpace of the search
        """

        self.cell = cell
        self.composition = self.cell.composition
        # position in composition space. Only used for phase diagram searches
        self.composition_vector = self.compute_composition_vector(
            composition_space)
        self.total_energy = None
        # the objective function value
        self.value = None
        # energy per atom
        self.epa = None
        # the fitness and relative fitness
        self.fitness = None
        self.relative_fitness = None
        # selection probability of this organism
        self.selection_prob = None
        # starting point of interval on number line of size selection_prob
        self.selection_loc = None
        # relative selection probability
        self.relative_selection_prob = None
        # starting point of interval of size relative_selection_prob
        self.relative_selection_loc = None
        # whether the organism is part of initial population or pool
        self.is_active = False
        # unique id number
        self._id = id_generator.make_id()
        # the name of the algorithm that created this organism
        self.made_by = maker
        # save the ids of parents if made by variation
        self.parents = None
        # Substrate related defaults
        # Number of atoms of the substrate in the interface
        self.n_sub = None
        # z coordinate used for selective dynamics
        self.sd_index=None
        # Save interface cell or just 2D cell
        self.interface_cell=None


    # This keeps the id (sort of) immutable by causing an exception to be
    # raised if the id is attempted to be set with org.id = some_id.
    @property
    def id(self):
        return self._id

    def compute_composition_vector(self, composition_space):
        """
        Returns the composition vector of the organism, as a numpy array.

        Args:
            composition_space: the CompositionSpace of the search.
        """

        if composition_space.objective_function == 'epa':
            return None
        elif composition_space.objective_function == 'pd':
            # make CompoundPhaseDiagram and PDAnalyzer objects
            pdentries = []
            for endpoint in composition_space.endpoints:
                pdentries.append(PDEntry(endpoint, -10))
            compound_pd = CompoundPhaseDiagram(pdentries,
                                               composition_space.endpoints)

            # transform the organism's composition
            transformed_entry = compound_pd.transform_entries(
                [PDEntry(self.composition, 10)], composition_space.endpoints)

            # get the transformed species and amounts
            if len(transformed_entry[0]) == 0:
                return None
            transformed_list = str(transformed_entry[0][0]).split()
            del transformed_list[0]
            popped = ''
            while popped != 'with':
                popped = transformed_list.pop()

            # separate the dummy species symbols from the amounts
            symbols = []
            amounts = []
            for entry in transformed_list:
                split_entry = entry.split('0+')
                symbols.append(split_entry[0])
                amounts.append(float(split_entry[1]))

            # make a dictionary mapping dummy species to amounts
            dummy_species_amounts = {}
            for i in range(len(symbols)):
                dummy_species_amounts[DummySpecie(symbol=symbols[i])] = \
                    amounts[i]

            # make Composition object with dummy species, get decomposition
            dummy_comp = Composition(dummy_species_amounts)
            decomp = compound_pd.get_decomposition(dummy_comp)

            # get amounts of the decomposition in terms of the (untransformed)
            # composition space endpoints
            formatted_decomp = {}
            for key in decomp:
                key_dict = key.as_dict()
                #print(key_dict)
                #comp = Composition(key_dict['entry']['composition'])
                comp = Composition(key_dict['composition'])
                formatted_decomp[comp] = decomp[key]

            # make the composition vector
            composition_vector = []
            # because the random organism creator shuffles the endpoints
            composition_space.endpoints.sort()
            for endpoint in composition_space.endpoints:
                if endpoint in formatted_decomp:
                    composition_vector.append(formatted_decomp[endpoint])
                else:
                    composition_vector.append(0.0)
            return np.array(composition_vector)

    def is_at_endpoint(self, composition_space):
        """
        Returns a boolean indicating whether the organism is located at an
        endpoint of the composition space.

        Args:
            composition_space: the CompositionSpace of the search
        """

        for endpoint in composition_space.endpoints:
            if self.composition.reduced_composition.almost_equals(endpoint):
                return True
        return False


class Cell(Structure):
    """
    Represents a cell. Provides additional functionality to the
    pymatgen.core.structure.Structure class.
    """
    def __init__(self, lattice, species, coords, charge=None,
                    validate_proximity=False, to_unit_cell=False,
                    coords_are_cartesian=False, site_properties=None,
                    labels = None, properties = None):
        """
        Takes all the attributes from Structure object and makes it into
        a Cell object.
        """
        super(Cell, self).__init__(lattice, species, coords, charge=charge,
                                        validate_proximity=validate_proximity,
                                        to_unit_cell=to_unit_cell,
                                        coords_are_cartesian=coords_are_cartesian,
                                        site_properties=site_properties)


    def rotate_to_principal_directions(self):
        """
        Rotates the cell into the principal directions. That is, lattice vector
        a is parallel to the Cartesian x-axis, lattice vector b lies in the
        Cartesian x-y plane and the z-component of lattice vector c is
        positive.

        Note: this method doesn't change the fractional coordinates of the
        sites. However, the Cartesian coordinates may be changed.
        """

        # rotate about the z-axis to align a vertically with the x-axis
        rotation = RotationTransformation(
            [0, 0, 1], 180 - (180/np.pi)*np.arctan2(
                    self.lattice.matrix[0][1],
                    self.lattice.matrix[0][0]))
        new_structure = rotation.apply_transformation(self)
        self.lattice = new_structure.lattice

        # rotate about the y-axis to make a parallel to the x-axis
        rotation = RotationTransformation(
                [0, 1, 0], (180/np.pi)*np.arctan2(
                    self.lattice.matrix[0][2],
                    self.lattice.matrix[0][0]))
        new_structure = rotation.apply_transformation(self)
        self.lattice = new_structure.lattice

        # rotate about the x-axis to make b lie in the x-y plane
        rotation = RotationTransformation(
                [1, 0, 0], 180 - (180/np.pi)*np.arctan2(
                    self.lattice.matrix[1][2],
                    self.lattice.matrix[1][1]))
        new_structure = rotation.apply_transformation(self)
        self.lattice = new_structure.lattice

        # make sure they are all pointing in positive directions
        if self.lattice.matrix[0][0] < 0:
            # rotate about y-axis to make a positive
            rotation = RotationTransformation([0, 1, 0], 180)
            new_structure = rotation.apply_transformation(self)
            self.lattice = new_structure.lattice
        if self.lattice.matrix[1][1] < 0:
            # rotate about x-axis to make b positive
            rotation = RotationTransformation([1, 0, 0], 180)
            new_structure = rotation.apply_transformation(self)
            self.lattice = new_structure.lattice
        if self.lattice.matrix[2][2] < 0:
            # mirror c across the x-y plane to make it positive
            # a and b
            a = self.lattice.matrix[0]
            b = self.lattice.matrix[1]
            # the components of c
            cx = self.lattice.matrix[2][0]
            cy = self.lattice.matrix[2][1]
            cz = -1*self.lattice.matrix[2][2]
            self.lattice = Lattice([a, b, [cx, cy, cz]])

    def rotate_c_parallel_to_z(self):
        """
        Rotates the cell such that lattice vector c is parallel to the
        Cartesian z-axis.

        Note: this method doesn't change the fractional coordinates of the
        sites. However, the Cartesian coordinates may be changed.
        """

        # rotate about the z-axis until c lies in the x-z plane
        rotation = RotationTransformation(
                [0, 0, 1], 180 - (180/np.pi)*np.arctan2(
                    self.lattice.matrix[2][1],
                    self.lattice.matrix[2][0]))
        new_structure = rotation.apply_transformation(self)
        self.lattice = new_structure.lattice

        # rotate about the y-axis to make c parallel to the z-axis
        rotation = RotationTransformation(
            [0, 1, 0], 180 - (180/np.pi)*np.arctan2(
                self.lattice.matrix[2][0],
                self.lattice.matrix[2][2]))
        new_structure = rotation.apply_transformation(self)
        self.lattice = new_structure.lattice

        # make sure c is pointing along the positive z-axis
        if self.lattice.matrix[2][2] < 0:
            # rotate 180 degrees about the x-axis
            rotation = RotationTransformation([1, 0, 0], 180)
            new_structure = rotation.apply_transformation(self)
            self.lattice = new_structure.lattice

    def translate_atoms_into_cell(self):
        """
        Translates all the atoms into the cell, so that their fractional
        coordinates are between 0 and 1.

        Precondition: all the atoms can fit inside the structure's lattice,
            and there is at least 0.001 of (fractional) extra space along
            each lattice vector.
        """

        # get the bounding box of the atoms, in fractional coordinates
        bounding_box = self.get_bounding_box(cart_coords=False)
        translation_vector = []

        # determine the needed shift along each lattice vector
        for i in range(3):
            if bounding_box[i][0] < 0.0:
                translation_vector.append(-1*(bounding_box[i][0]) + 0.001)
            elif bounding_box[i][1] >= 1.0:
                translation_vector.append(-1*(bounding_box[i][1] + 0.001) +
                                          1.0)
            else:
                translation_vector.append(0.0)

        # do the shift
        site_indices = [i for i in range(len(self.sites))]
        self.translate_sites(site_indices, translation_vector,
                             frac_coords=True, to_unit_cell=False)

    def reduce_cell(self):
        """
        Applies Niggli cell reduction to cell.

        Returns a boolean indicated whether cell reduction was successful.

        Note: pymatgen's Niggli cell reduction algorithm sometimes moves the
            atoms' relative positions a little (I've seen up to 0.5 A...)
        """
        # get the reduced structure
        try:  # sometimes pymatgen's reduction routine fails
            reduced_structure = self.get_reduced_structure()
        except:
            return False

        # modify the cell to correspond to the reduced structure
        rcartesian_coords = reduced_structure.cart_coords
        rspecies = reduced_structure.species
        self.lattice = reduced_structure.lattice
        site_indices = []
        for i in range(len(self.sites)):
            site_indices.append(i)
        self.remove_sites(site_indices)
        for i in range(len(rcartesian_coords)):
            self.append(rspecies[i], rcartesian_coords[i],
                        coords_are_cartesian=True)

        # rotate the cell into the principal directions
        self.rotate_to_principal_directions()
        return True

    def reduce_sheet_cell(self, geometry, constraints):
        """
        Applies Niggli cell reduction to a sheet cell. The idea is to make
        lattice vector c vertical and add lots of vertical vacuum so that the
        standard reduction algorithm only changes the a and b lattice vectors.

        Returns a boolean indicating whether cell reduction was successful.

        Args:
            geometry: the Geometry of the search

            constraints: the Constraints of the search
        """
        # pad the cell with 100 Angstroms of vertical vacuum
        geometry.pad(self, padding=100)

        # reduce the padded cell
        successful_reduction = self.reduce_cell()

        # unpad the reduced cell
        geometry.unpad(self, "", constraints)

        return successful_reduction

    def get_bounding_box(self, cart_coords=True):
        """
        Returns the smallest and largest coordinates in each dimension of all
        the sites in a cell, as a list of three lists:

            [[min, max], [min, max], [min, max]]

        where the first inner list contains data for the Cartesian x-coordinate
        or lattice vector a, the second inner list contains data for the
        Cartesian y-coordinate or lattice vector b, and the third inner list
        contains data for the Cartesian z-coordinate or lattice vector c.

        Args:
            cart_coords: whether to give the result in Cartesian or fractional
                coordinates
        """

        if cart_coords:
            coords = self.cart_coords
        else:
            coords = self.frac_coords

        # find the largest and smallest coordinates in each dimension
        minx = np.inf
        maxx = -np.inf
        miny = np.inf
        maxy = -np.inf
        minz = np.inf
        maxz = -np.inf
        for coord in coords:
            if coord[0] < minx:
                minx = coord[0]
            if coord[0] > maxx:
                maxx = coord[0]
            if coord[1] < miny:
                miny = coord[1]
            if coord[1] > maxy:
                maxy = coord[1]
            if coord[2] < minz:
                minz = coord[2]
            if coord[2] > maxz:
                maxz = coord[2]
        return [[minx, maxx], [miny, maxy], [minz, maxz]]

    def surface_area(cell):
        """
        Calculates the surface area of the Cell
        """
        m = cell.lattice.matrix
        return np.linalg.norm(np.cross(m[0], m[1]))


class OffspringGenerator:
    """
    This class handles generating offspring organisms from the pool and the
    variations. Basically a place for the make_offspring_organism method to
    live.
    """
    def __init__(self,pd, composition_space):
        self.pd = pd
        self.composition_space = composition_space
    def make_offspring_organism(self, random, pool, variations, geometry,
                                id_generator, whole_pop, developer,
                                redundancy_guard, composition_space,
                                constraints, num_mating, num_mutation):
        """
        Returns a developed, non-redundant, unrelaxed offspring organism
        generated with one of the variations.

        Args:
            random: a copy of Python's PRNG

            pool: the Pool

            variations: list of all the Variations used in this search

            geometry: the Geometry of the search

            id_generator: the IDGenerator used to assign id numbers to all
                organisms

            whole_pop: list containing copies of the organisms to check for
                redundancy

            developer: the Developer of the search

            redundancy_guard: the RedundancyGuard of the search

            composition_space: the CompositionSpace of the search

            constraints: the Constraints of the search

        Description:

            1. Randomly selects one of the Variations (based on the fraction
                values of the Variations) with which to make an offspring
                Organism.

            2. Tries to generate a developed, non-redundant, unrelaxed
                offspring Organism with the selected Variation.

            3. If no success after 1000 attempts, randomly selects a different
                Variation and tries to make an offspring with that one.

            4. If no success after trying all the Variations, cycles through
                them again.
        """

        # holds the variations that have been tried
        tried_variations = []
        # the maximum number of times to try making a valid offspring with a
        # given variation before giving up and trying a different variation
        max_num_tries = 10000

        while True:
            #print('general.py line 522')
            variation = self.select_variation(random, tried_variations,
                                              variations)
           

            num_tries = 0
            candidates = []
            if variation.name == 'permutation':
                n = num_mutation
            if variation.name == 'mating':
                n = num_mating
            if variation.name == 'structure mutation':
                n = num_mutation
            if variation.name == 'number of atoms mutation':
                n = num_mutation
            print('-----------------------------------------------------------------------------------------')
            print(f'Performing {variation.name} pool size of N = {n}...')
            print('-----------------------------------------------------------------------------------------')
            while num_tries < max_num_tries and len(candidates) < n:
                #candidates = variation.looped(
                   # pool, random, geometry, constraints, id_generator,
                    #composition_space,40)
                offspring = variation.do_variation(
                    pool, random, geometry, constraints, id_generator,
                    composition_space)
                #epas,compositions = variation.predict_batches(candidates)
                #offspring,score = variation.calculate_scores(candidates, compositions, epas, self.pd, self.composition_space)
                
                if developer.develop(
                        offspring, composition_space, constraints, geometry,
                        pool) and (redundancy_guard.check_redundancy(
                            offspring, whole_pop, geometry) is None):
                                    
                    candidates.append(offspring)
                    #print('candidate sizes:',len(o.cell)
                else:
                    num_tries = num_tries + 1
            #print('candidate sizes:',[len(o.cell) for o in candidates])
            candidate_groups = defaultdict(list)
            for group in candidates:
                candidate_groups[len(group.cell)].append(group)

            grouped_candidates = [candidate_groups[key] for key in sorted(candidate_groups)]
            candidate_pool = []
            for candidates in grouped_candidates:
                energies,compositions = variation.predict_batches(candidates)
                offspring,score = variation.calculate_scores(candidates, compositions, energies, self.pd, self.composition_space)
                candidate_pool.append([offspring,score])
            
            
            candidate_pool = [org for org in candidate_pool if org[0] != None]
            if len(candidate_pool) == 0:
                return [None]
            offspring = random.sample(candidate_pool, k = 1)
            selected_offsprings = [offspring[0]]
            for candidate in candidate_pool:
                if candidate[1] <= 0.1:
                    selected_offsprings.append(candidate)
            tried_variations.append(variation)
            #print(selected_offsprings)
            print(f'        ---> selecting candidates with comps {[i[0].cell.composition for i in selected_offsprings]} and scores {[i[1] for i in selected_offsprings]}')
            return [offspring_organism[0] for offspring_organism in selected_offsprings] #offspring[0][0]
            # if we've tried all the variations without success, then cycle
            # through them again
            if len(tried_variations) == len(variations):
                tried_variations = []

    def select_variation(self, random, tried_variations, variations):
        """
        Selects a variation that hasn't been tried yet based on their selection
        probabilities.

        Args:
            random: a copy of Python's PRNG

            tried_variations: list of Variations that have already been
                unsuccessfully tried

            variations: list of all the Variations used in this search
        """

        while True:
            rand = random.random()
            lower_bound = 0
            for i in range(len(variations)):
                if rand > lower_bound and rand <= (
                        lower_bound + variations[i].fraction) and \
                        variations[i] not in tried_variations:
                    return variations[i]
                else:
                    lower_bound = lower_bound + variations[i].fraction


class SelectionProbDist(object):
    """
    Defines the probability distribution over the fitnesses of the organisms
    in the pool that determines their selection probabilities.

    Comprised of two numbers: the number of organisms to select from, and the
    selection power.
    """

    def __init__(self, selection_params, pool_size):
        """
        Makes a SelectionProbDist, and sets default parameter values if
        necessary.

        Args:
            selection_params: the parameters defining the distribution, as a
                dictionary

            pool_size: the size of the Pool (how many organisms it contains)
        """

        # default values
        self.default_num_parents = pool_size
        self.default_power = 1

        # if entire Selection block was left blank or set to default
        if selection_params in (None, 'default'):
            self.num_parents = self.default_num_parents
            self.power = self.default_power
        else:
            # check the num_parents parameter
            if 'num_parents' not in selection_params:
                self.num_parents = self.default_num_parents
            elif selection_params['num_parents'] in (None, 'default'):
                self.num_parents = self.default_num_parents
            # check that the given number isn't larger than the pool size
            elif selection_params['num_parents'] > pool_size:
                self.num_parents = self.default_num_parents
            else:
                self.num_parents = selection_params['num_parents']

            # check the power parameter
            if 'power' not in selection_params:
                self.power = self.default_power
            elif selection_params['power'] in (None, 'default'):
                self.power = self.default_power
            else:
                self.power = selection_params['power']


class CompositionSpace(object):
    """
    Represents the composition space to be searched by the algorithm.
    """

    def __init__(self, endpoints, sub_search=False):
        """
        Makes a CompositionSpace, which is list of
        pymatgen.core.composition.Composition objects.

        Args:
            endpoints: the list of compositions, as strings (e.g., ["Al2O3"])

            sub_search (bool): whether it is a interface geometry
        """

        for i in range(len(endpoints)):
            endpoints[i] = Composition(endpoints[i])
            endpoints[i] = endpoints[i].reduced_composition

        self.endpoints = endpoints

        # If it is substrate search, set these defaults and new_obj_fn
        self.sub_search = sub_search

        # objective function lives here
        self.objective_function = self.infer_objective_function()

        # the center of the composition space
        self.center = self.get_center()

        # the distance from an endpoint to the center
        self.max_dist_from_center = (float(len(self.endpoints) - 1))/float(len(
            self.endpoints))

        species_dict = {}
        elems = self.get_all_elements()
        # Specify species A, B, C always in increasing atomic number
        sorted_elems = sorted(elems, key=lambda x: x.number)
        species_dict['specie_A'] = sorted_elems[0]
        if len(elems) > 1:
            species_dict['specie_B'] = sorted_elems[1]
        if len(elems) > 2:
            species_dict['specie_C'] = sorted_elems[2]
        self.species_dict = species_dict


    def infer_objective_function(self):
        """
        Infers the objective function (energy per atom or phase diagram) based
        on the composition space.

        Returns either 'epa' or 'pd'.

        For substrate search, epa is the energy of formation of adsorption
        on substrate.

        EDIT: In substrate search, the formationn energy calculated using
        chemical potentials needs to be used in place of 'epa'. Also, for a
        composition range (2 or more endpoints), 'pd' cannot be used. Because
        the total energy also includes substrate energy, while composiiton is
        that of only 2D system.

        So, eliminating 'pd' completely for substrate search while using
        chemical potentials.

        TODO: Include total_adsorption_energy and formation_energy as attributes
        which activate during substrate search. Depending on the type of
        energy chosen, use 'pd' or 'epa' type objective_function respectively.
        """

        if len(self.endpoints) == 1:
            obj = 'epa'
        else:
            for point in self.endpoints:
                for next_point in self.endpoints:
                    if not point.almost_equals(next_point, 0.0, 0.0):
                        obj = 'pd'

        # For substrate search, use 'epa' here. However, chemical potential
        # based surface free energy is used as objective function, although,
        # it reads 'epa'. See energy_calculators.do_energy_calculation for more
        if self.sub_search:
            obj = 'epa'

        return obj

    def get_center(self):
        """
        Returns the center of the composition space, as a numpy array.
        """

        center_vect = []
        for _ in range(len(self.endpoints)):
            center_vect.append(1/len(self.endpoints))
        return np.array(center_vect)

    def get_all_elements(self):
        """
        Returns a list of all the elements
        (as pymatgen.core.periodic_table.Element objects) that are in the
        composition space.
        """

        # get each element type from the composition_space object
        elements = []
        for point in self.endpoints:
            elements = elements + point.elements

        # remove duplicates
        elements = list(set(elements))
        return elements

    def get_all_pairs(self):
        """
        Returns all possible pairs of elements in the composition space, as
        list of strings, where each string contains the symbols of two
        elements, separated by a space. Does not include self-pairs
        (e.g., "Cu Cu").
        """

        elements = self.get_all_elements()
        if len(elements) == 1:
            return []
        pairs = []
        for i in range(0, len(elements) - 1):
            for j in range(i + 1, len(elements)):
                pairs.append(str(elements[i].symbol + " " +
                                 elements[j].symbol))
        return pairs

    def get_all_swappable_pairs(self):
        """
        Computes all pairs of elements in the the composition space that are
        allowed to be swapped based on their electronegativities. Only pairs
        whose electronegativities differ by 1.1 or less can be swapped.

        Returns a list of strings, where each string contains the symbols of
        two elements, separated by a space. Does not include self-pairs
        (e.g., "Cu Cu").
        """

        all_pairs = self.get_all_pairs()
        allowed_pairs = []
        for pair in all_pairs:
            element1 = Element(pair.split()[0])
            element2 = Element(pair.split()[1])
            diff = abs(element1.X - element2.X)
            if diff <= 1.1:
                allowed_pairs.append(pair)
        return allowed_pairs


class CompositionFitnessWeight(object):
    """
    Defines how the weight given to the composition fitness of organisms in
    phase diagram searches is computed.
    """

    def __init__(self, comp_fitness_params):
        """
        Makes a CompositionFitnessWeight, and sets default parameters if
        necessary.

        Args:
            comp_fitness_params: a dictionary of parameters
        """

        # default values
        self.default_max_weight = 0.5
        self.default_power = 1

        # if entire CompositionFitnessWeight block was left blank or set to
        # default
        if comp_fitness_params in (None, 'default'):
            self.max_weight = self.default_max_weight
            self.power = self.default_power
        else:
            # check the max_weight parameter
            if 'max_weight' not in comp_fitness_params:
                self.max_weight = self.default_max_weight
            elif comp_fitness_params['max_weight'] in (None, 'default'):
                self.max_weight = self.default_max_weight
            else:
                self.max_weight = comp_fitness_params['max_weight']

            # check the power parameter
            if 'power' not in comp_fitness_params:
                self.power = self.default_power
            elif comp_fitness_params['power'] in (None, 'default'):
                self.power = self.default_power
            else:
                self.power = comp_fitness_params['power']


class StoppingCriteria(object):
    """
    Defines when the search should be stopped.
    """

    def __init__(self, stopping_parameters, composition_space):
        """
        Makes a StoppingCriteria, and sets default parameter values if
        necessary.

        Args:
            stopping_parameters: a dictionary of parameters

            composition_space: the CompositionSpace of the search
        """

        # set the defaults
        if len(composition_space.endpoints) == 1:
            # for fixed composition searches
            self.default_num_energy_calcs = 800
        elif len(composition_space.endpoints) == 2:
            # for binary phase diagram searches
            self.default_num_energy_calcs = 1000
        elif len(composition_space.endpoints) == 3:
            # for ternary phase diagram searches
            self.default_num_energy_calcs = 3000
        elif len(composition_space.endpoints) == 4:
            # for quaternary phase diagram searches
            self.default_num_energy_calcs = 6000

        # the energy per atom at which to stop
        self.default_epa_achieved = None
        # the Cell at which to stop when found
        self.default_found_cell = None
        # whether or not the stopping criteria are satisfied
        self.are_satisfied = False
        # to keep track of how many energy calculations have been done
        self.calc_counter = 0

        # set defaults if stopping_parameters equals 'default' or None
        if stopping_parameters in (None, 'default'):
            self.num_energy_calcs = self.default_num_energy_calcs
            self.epa_achieved = self.default_epa_achieved
            self.found_cell = self.default_found_cell
        # check each keyword to see if it's been included
        else:
            # value achieved
            if 'epa_achieved' in stopping_parameters:
                if stopping_parameters['epa_achieved'] in (None, 'default'):
                    self.epa_achieved = self.default_epa_achieved
                elif composition_space.objective_function == 'epa':
                    self.epa_achieved = stopping_parameters['epa_achieved']
            else:
                self.epa_achieved = self.default_epa_achieved

            # found structure
            if 'found_structure' in stopping_parameters:
                if stopping_parameters['found_structure'] in (None, 'default'):
                    self.found_cell = self.default_found_cell
                else:
                    self.path_to_structure_file = stopping_parameters[
                        'found_structure']
                    self.found_cell = Cell.from_file(
                            stopping_parameters['found_structure'])
            else:
                self.found_cell = self.default_found_cell

            # number energy calculations
            if 'num_energy_calcs' in stopping_parameters:
                if stopping_parameters['num_energy_calcs'] in (None,
                                                               'default'):
                    if self.epa_achieved is None and \
                            self.found_cell is None:
                        self.num_energy_calcs = self.default_num_energy_calcs
                    else:
                        self.num_energy_calcs = None
                else:
                    self.num_energy_calcs = stopping_parameters[
                        'num_energy_calcs']
            elif self.epa_achieved is None and self.found_cell is None:
                self.num_energy_calcs = self.default_num_energy_calcs
            else:
                self.num_energy_calcs = None

    def update_calc_counter(self):
        """
        If num_energy_calcs stopping criteria is being used, increments calc
        counter and updates are_satisfied if necessary.
        """

        if self.num_energy_calcs is not None:
            self.calc_counter += 1
            if self.calc_counter >= self.num_energy_calcs:
                self.are_satisfied = True

    def check_organism(self, organism, redundancy_guard, geometry):
        """
        If value_achieved or found_structure stopping criteria are used, checks
        if the relaxed organism satisfies them, and updates are_satisfied.

        Args:
            organism: a relaxed Organism whose value has been computed

            redundancy_guard: the RedundancyGuard of the search

            geometry: the Geometry of the search
        """

        # check the objective function value if needed
        if self.epa_achieved is not None:
            if organism.epa <= self.epa_achieved:
                self.are_satisfied = True

        # check the structure if needed
        if self.found_cell is not None:
            if geometry.shape == 'cluster':
                mol1 = Molecule(organism.cell.species,
                                organism.cell.cart_coords)
                mol2 = Molecule(self.found_cell.species,
                                self.found_cell.cart_coords)
                self.are_satisfied = \
                    redundancy_guard.molecule_matcher.fit(mol1, mol2)
            else:
                self.are_satisfied = \
                    redundancy_guard.structure_matcher.fit(organism.cell,
                                                           self.found_cell)


class DataWriter(object):
    """
    For writing useful data to a file in the course of a search.
    """

    def __init__(self, garun_dir, composition_space, sub_search=False):
        """
        Makes a DataWriter.

        Args:
            garun_dir: the path to the garun directory where run_data file is
            to be written

            composition_space: the CompositionSpace of the search

            sub_search (bool): whether it is interface geometry search
        """

        self.file_path = garun_dir + '/run_data'
        self.sub_search = sub_search
        with open(self.file_path, 'a') as data_file:
            data_file.write('Composition space endpoints: ')
            for endpoint in composition_space.endpoints:
                data_file.write(' {}'.format(
                    endpoint.reduced_formula.replace(' ', '')))
            data_file.write('\n\n')
            if not sub_search:
                data_file.write('id\t\t composition\t total energy\t\t '
                                'epa\t\t\t num calcs\t best value\n\n')
            else:
                data_file.write('id\t comp\t n-2D\t n-sub\t surface area\t '
                                'total energy\t  epa\t\t num calcs\t best value\n\n')

        self.genes_file = garun_dir + '/genes_data'
        with open(self.genes_file, 'a') as genes:
            genes.write('id\t parents id(s)\t maker\n\n')

    def write_data(self, organism, num_calcs, progress):
        """
        Writes data to self.file_path in the format:

            id    composition    total energy    epa    num calcs    best value

        Args:
            organism: the Organism whose data to write

            num_calcs: the number of calculations finished so far,
                including failures

            progress: for fixed-composition searches, the best epa seen by the
                algorithm so far. For phase diagram searches, the area/volume
                of the convex hull, or None if no convex hull could be
                constructed.
        """
        # if progress is None due to some error in get_convex_hull_area
        if progress is None:
            progress = 0.0

        # determine how many tabs to use after the composition
        formula = organism.composition.formula.replace(' ', '')
        if len(formula) > 8:
            format_string = '{0}\t\t {1}\t {2:.6f}\t\t {3:.6f}\t\t {4}\t\t'
        else:
            format_string = '{0}\t\t {1}\t\t {2:.6f}\t\t {3:.6f}\t\t {4}\t\t'
        if self.sub_search: # if this is a substrate search
            format_string = '{0}\t {1}\t {2}\t {3}\t {4:.6f}\t {5:.6f}\t {6:.6f}\t {7}\t\t'
            num_twod = len(organism.cell.sites)
            ab = organism.cell.lattice.matrix[:2]
            org_surface_area = np.linalg.norm(np.cross(ab[0], ab[1]))


        # determine what to write for the progress
        if self.sub_search:
            format_string = format_string + ' {8:.6f}\n'
        else:
            if progress is None:
                format_string = format_string + ' None\n'
            else:
                format_string = format_string + ' {5:.6f}\n'

        # write the line to the file
        with open(self.file_path, 'a') as data_file:
            if self.sub_search:
                data_file.write(format_string.format(
                    organism.id, formula, num_twod, organism.n_sub,
                    org_surface_area, organism.total_energy, organism.epa,
                    num_calcs, progress))
            else:
                print('data_writer types organism.id, formula, organism.total_energy, organism.epa,num_calcs,progress:',[organism.id,formula,organism.total_energy,organism.epa,num_calcs,progress])
                data_file.write(format_string.format(
                    organism.id, formula, organism.total_energy, organism.epa,
                    num_calcs, progress))

        # write genes_data file
        self.write_genes_file(organism)

    def write_genes_file(self, organism):
        """
        Writes the maker of every organism and including parents for an
        offspring organism to a file "genes_data" in the garun folder.

        Args:
            organism: the Organism whose data to write
        """
        with open(self.genes_file, 'a') as genes:
            genes.write('{0}\t {1}\t {2}\n'.format(
                            organism.id, organism.parents, organism.made_by))
