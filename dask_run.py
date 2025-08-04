from smart_gasp.population import organism_creators
from smart_gasp.population import population
from smart_gasp.general import objects_maker
from smart_gasp import parameters_printer
from smart_gasp.evolution import variations
#from gasp import interface
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import numpy as np
from collections import Counter
# change worker unresponsive time to 3h (Assuming max elapsed time for one calc)
import dask
import dask.distributed
import copy
import threading
import random
import sys
import yaml
import os
import datetime
from time import sleep
from dask.distributed import Client
import dask
import dask.distributed
from smart_gasp import general
#from gasp import general
import pickle
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PDEntry
class SMART_GASP():
    def __init__(self):
        # get dictionaries from the input file (in yaml format)
        if len(sys.argv) < 2:
            print('No input file given.')
            print('Quitting...')
            quit()
        else:
            input_file = os.path.abspath(sys.argv[1])

        try:
            with open(input_file, 'r') as f:
                self.parameters = yaml.load(f, Loader=yaml.FullLoader)
        except:
            print('Error reading input file.')
            print('Quitting...')
            quit()

        # make the objects needed by the algorithm
        self.whole_pop = []
        self.num_finished_calcs = 0
        self.objects_dict = objects_maker.make_objects(self.parameters)
        self.geometry = self.objects_dict['geometry']
        self.run_dir_name = self.objects_dict['run_dir_name']
        self.organism_creators = self.objects_dict['organism_creators']
        self.num_calcs_at_once = self.objects_dict['num_calcs_at_once']
        self.composition_space = self.objects_dict['composition_space']
        self.constraints = self.objects_dict['constraints']
        self.developer = self.objects_dict['developer']
        self.redundancy_guard = self.objects_dict['redundancy_guard']
        self.stopping_criteria = self.objects_dict['stopping_criteria']
        self.energy_calculator = self.objects_dict['energy_calculator']
        self.pool = self.objects_dict['pool']
        self.variations = self.objects_dict['variations']
        self.id_generator = self.objects_dict['id_generator']
        self.job_specs = self.objects_dict['job_specs']
        self.bool = True
        self.garun_dir = str(os.getcwd()) + '/' + self.run_dir_name
        garun_dir = self.garun_dir
        if os.path.isdir(garun_dir):
            print('Directory {} already exists'.format(garun_dir))
            time = datetime.datetime.now().time()
            date = datetime.datetime.now().date()
            current_date = str(date.month) + '_' + str(date.day) + '_' + \
                str(date.year)
            current_time = str(time.hour) + '_' + str(time.minute) + '_' + \
                str(time.second)
            garun_dir += '_' + current_date + '_' + current_time
            print('Setting the run directory to {}'.format(garun_dir))
        os.mkdir(garun_dir)
        os.chdir(garun_dir)
        os.mkdir(garun_dir + '/temp')
        parameters_printer.print_parameters(self.objects_dict,
                                            lat_match_dict=None)
        data_writer = general.DataWriter(garun_dir,
                                self.composition_space, sub_search=False)
        """
        cluster_job = SLURMCluster(cores=self.job_specs['cores'],
                                   memory=self.job_specs['memory'],
                                   project=self.job_specs['project'],
                                   queue=self.job_specs['queue'],
                                   interface=self.job_specs['interface'],
                                   walltime=self.job_specs['walltime'],
                                   job_extra=self.job_specs['job_extra'])
        cluster_job.scale(self.num_calcs_at_once) # number of parallel jobs
        """
        self.client  = Client(cluster_job)
        self.client.upload_file('/blue/hennig/sam.dong/SMART_gasp/GASP-python/smart_gasp.zip')
        self.futures = []
        self.initial_pop = None
        
    def initial_population(self):


        num_finished_calcs = 0
        self.initial_pop = population.InitialPopulation(self.run_dir_name)

        while self.bool:
            for creator in self.organism_creators:
                print('------------------------------------------------------------------------------------------------')
                print('Making {} organisms with {}'.format(creator.number,
                                                           creator.name))
                while not creator.is_finished and not self.stopping_criteria.are_satisfied:
                    working_jobs = len([i for i, f in enumerate(self.futures) \
                                                        if not f.done()])
                    if working_jobs < self.num_calcs_at_once:
                        new_organism = creator.create_organism(
                            self.id_generator, self.composition_space, self.constraints, random)
                        while new_organism is None and not creator.is_finished:
                            new_organism = creator.create_organism(
                                self.id_generator, self.composition_space, self.constraints, random)
                        if new_organism is not None:
                            self.geometry.unpad(new_organism.cell, new_organism.n_sub,
                                                                        self.constraints)
                            if self.developer.develop(new_organism, self.composition_space,
                                                 self.constraints, self.geometry, self.pool):
                                redundant_organism = self.redundancy_guard.check_redundancy(
                                    new_organism, self.whole_pop, self.geometry)
                                if redundant_organism is None:  # no redundancy
                                    self.whole_pop.append(copy.deepcopy(new_organism))
                                    relaxed_organism = self.energy_calculator.do_energy_calculation(new_organism, self.composition_space)
                                    if relaxed_organism is not None:
                                        self.geometry.unpad(relaxed_organism.cell,
                                            relaxed_organism.n_sub, self.constraints)
                                        self.initial_pop.add_organism(relaxed_organism, self.composition_space)
                                        self.num_finished_calcs += 1
                                        self.stopping_criteria.update_calc_counter()
                                        self.stopping_criteria.check_organism(relaxed_organism, self.redundancy_guard,self.geometry)
                                        self.whole_pop.append(relaxed_organism)
                                        progress = \
                                        self.initial_pop.get_progress(
                                            self.composition_space)
                                    else:
                                        new_organism = creator.create_organism(self.id_generator,self.composition_space,
                                                self.constraints, random)
            try:
                self.pool.add_initial_population(self.initial_pop, self.composition_space)
                return self.pool
            except:
                print('*** not all endpoints present, generating more structures ***')
                self.objects_dict = objects_maker.make_objects(self.parameters)
                self.organism_creators = self.objects_dict['organism_creators']
                pass

        return self.pool


    def perform_variations(self,pool):

        i = 0
        scores = []
        self.num_finished_calcs = 0
        while self.num_finished_calcs < 1000:
            offspring_generator = general.OffspringGenerator(self.pool.compound_pd, self.composition_space)
            unrelaxed_offsprings = offspring_generator.make_offspring_organism(
            random, self.pool, self.variations, self.geometry, self.id_generator, self.whole_pop,
            self.developer, self.redundancy_guard, self.composition_space, self.constraints)
            for unrelaxed_offspring in unrelaxed_offsprings:
                if unrelaxed_offspring is not None:
                    self.whole_pop.append(copy.deepcopy(unrelaxed_offspring))
                    self.geometry.pad(unrelaxed_offspring.cell)
                    relaxed_offspring = self.energy_calculator.do_energy_calculation(unrelaxed_offspring, self.composition_space)
                    if relaxed_offspring is not None:
                        if self.developer.develop(relaxed_offspring,
                                    self.composition_space,
                                    self.constraints, self.geometry, self.pool):
                            redundant_organism = self.redundancy_guard.check_redundancy(relaxed_offspring, self.pool.to_list(), self.geometry)
                            if redundant_organism is not None:
                                if redundant_organism.epa > relaxed_offspring.epa:
                                    self.pool.replace_organism(redundant_organism,
                                        relaxed_offspring,
                                        self.composition_space)
                                    self.num_finished_calcs += 1
                                    self.pool.compute_fitnesses()
                                    self.pool.compute_selection_probs()
                                    progress = self.pool.get_progress(self.composition_space)
                            else:
                                self.num_finished_calcs += 1
                                self.stopping_criteria.update_calc_counter()
                                self.pool.add_organism(relaxed_offspring, self.composition_space)
                                self.whole_pop.append(relaxed_offspring)
                                removed_org = self.pool.queue.pop()
                                removed_org.is_active = False
                                self.whole_pop.append(relaxed_offspring)
                                self.pool.compute_fitnesses()
                                self.pool.compute_selection_probs()
                                progress = self.pool.get_progress(self.composition_space)
                                remove_bool = np.random.randint(0,2)
                                if len(self.pool.queue) > int(self.pool.size):
                                    if remove_bool == 0:
                                        print([f'Organism ({o.id},{o.cell.composition}): {o.value}' for o in self.pool.queue])
                                        print(f'     ---> Removing worst Organism ({self.pool.queue[-1].id}, {self.pool.queue[-1].cell.composition}) from population pool with value {self.pool.queue[-1].value}')
                                        self.pool.queue.pop()
                                    else:
                                        comp_count = Counter(o.cell.composition for o in self.pool.queue)
                                        most_common_comp, _ = comp_count.most_common(1)[0]
                                        most_common_candidates = [(i,o) for i,o in enumerate(self.pool.queue) if o.cell.composition == most_common_comp]
                                        indices, _ = max(most_common_candidates, key = lambda x: x[1].value)
                                        print(f'     ---> Removing most common Organism ({self.pool.queue[indices].id}, {self.pool.queue[indices].cell.composition}) from population pool with value {self.pool.queue[indices].value}')
                                        del self.pool.queue[indices]
                                        print([f'Organism ({o.id},{o.cell.composition}): {o.value}' for o in self.pool.queue])

                    else:
                        pass
                else:
                    pass



def main():
    smart_ga = SMART_GASP()

    pool = smart_ga.initial_population()
    smart_ga.perform_variations(pool)
    print('done!')
if __name__ == "__main__":
    main()

