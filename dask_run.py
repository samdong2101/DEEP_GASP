from deep_gasp.population import organism_creators
from deep_gasp.population import population
from deep_gasp.general import objects_maker
from deep_gasp import parameters_printer
from deep_gasp.evolution import variations
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
#from dask_jobqueue import SLURMCluster
#from dask.distributed import Client
import numpy as np
from collections import Counter
#import dask
#import dask.distributed
import copy
import threading
import random
import sys
from pathlib import Path
import yaml
import os
import datetime
from datetime import datetime
from time import sleep
#from dask.distributed import Client
#import dask
#import dask.distributed
from deep_gasp import general
import pickle
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.analysis.phase_diagram import PDEntry
import psutil
import logging
import signal

class DEEP_GASP():
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
        self.num_mating = self.parameters['TBS']['num_mating']
        self.num_mutation = self.parameters['TBS']['num_mutation']
        self.total_calcs = self.parameters['StoppingCriteria']['num_energy_calcs']
        garun_dir = self.garun_dir

        elements_str = "_".join(str(el) for comp in self.composition_space.endpoints for el in comp.elements)
        now = str(datetime.now()).replace(' ','_')
        garun_dir += '_' + elements_str + '_' + now
        os.mkdir(garun_dir)
        os.chdir(garun_dir)
        os.mkdir(garun_dir + '/temp')
        parameters_printer.print_parameters(self.objects_dict,
                                            lat_match_dict=None)
        data_writer = general.DataWriter(garun_dir,
                                self.composition_space, sub_search=False)
        self.futures = []
        self.initial_pop = None
        logging.basicConfig(level=logging.INFO, format = "%(asctime)s - %(message)s")
        log_dir = Path(os.getcwd())
        log_dir.mkdir(parents=True, exist_ok=True)
        job_id = os.environ.get("SLURM_JOB_ID", "nojobid")
        stdout_path = log_dir / f"job_{job_id}.log"
        stderr_path = log_dir / f"err_{job_id}.log"
        sys.stdout = open(stdout_path, "w")
        sys.stderr = open(stderr_path, "w")

    def initial_population(self):


        num_finished_calcs = 0
        self.initial_pop = population.InitialPopulation(self.run_dir_name)
        signal.signal(signal.SIGALRM, self.timeout_handler)
        while self.bool:
            for creator in self.organism_creators:
                print('------------------------------------------------------------------------------------------------')
                print(f' ---> searching within composition space: {self.composition_space.endpoints}')
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
                            #if self.developer.develop(new_organism, self.composition_space,
                                                 #self.constraints, self.geometry, self.pool):
                            redundant_organism = self.redundancy_guard.check_redundancy(
                                new_organism, self.whole_pop, self.geometry)
                            if redundant_organism is None:  # no redundancy
                                self.whole_pop.append(copy.deepcopy(new_organism))
                                signal.alarm(60)
                                try:
                                    relaxed_organism = self.energy_calculator.do_energy_calculation(new_organism, self.composition_space)
                                except Exception as e:
                                    print(' ---> Convergence Failure... continuing')
                                    continue
                                finally:
                                    signal.alarm(0)
                                self.log_memory_usage("Inititial population - energy calculation on organism {new_organism.id}")
                                if relaxed_organism is not None:
                                    self.geometry.unpad(relaxed_organism.cell,
                                        relaxed_organism.n_sub, self.constraints)
                                    self.initial_pop.add_organism(relaxed_organism, self.composition_space)
                                    self.ga_logger(relaxed_organism,add = True, init_pop = True)
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
        signal.signal(signal.SIGALRM, self.timeout_handler)
        while self.num_finished_calcs < self.total_calcs:
            offspring_generator = general.OffspringGenerator(self.pool.compound_pd, self.composition_space)
            unrelaxed_offsprings = offspring_generator.make_offspring_organism(
            random, self.pool, self.variations, self.geometry, self.id_generator, self.whole_pop,
            self.developer, self.redundancy_guard, self.composition_space, self.constraints,
            self.num_mating, self.num_mutation)
            for unrelaxed_offspring in unrelaxed_offsprings:
                if unrelaxed_offspring is not None:
                    self.whole_pop.append(copy.deepcopy(unrelaxed_offspring))
                    self.geometry.pad(unrelaxed_offspring.cell)
                    signal.alarm(60)
                    try:
                        relaxed_offspring = self.energy_calculator.do_energy_calculation(unrelaxed_offspring, self.composition_space)
                    except Exception as e:
                        print(' ---> Convergence Failure... continuing')
                        continue
                    finally:
                        signal.alarm(0)
                    self.log_memory_usage("Offspring - energy calculation on organism {unrelaxed_offspring.id}")
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
                                    self.ga_logger(relaxed_offspring,add = True,pop_pool = True)
                                    self.num_finished_calcs += 1
                                    self.pool.compute_fitnesses()
                                    self.pool.compute_selection_probs()
                                    progress = self.pool.get_progress(self.composition_space)
                            else:
                                self.num_finished_calcs += 1
                                self.stopping_criteria.update_calc_counter()
                                self.pool.add_organism(relaxed_offspring, self.composition_space)
                                self.ga_logger(relaxed_offspring,add = True,pop_pool = True)
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
                                        self.ga_logger(self.pool.queue[-1],remove = True,pop_pool = True)
                                        self.pool.queue.pop()
                                    else:
                                        comp_count = Counter(o.cell.composition for o in self.pool.queue)
                                        most_common_comp, _ = comp_count.most_common(1)[0]
                                        most_common_candidates = [(i,o) for i,o in enumerate(self.pool.queue) if o.cell.composition == most_common_comp]
                                        indices, _ = max(most_common_candidates, key = lambda x: x[1].value)
                                        print(f'     ---> Removing most common Organism ({self.pool.queue[indices].id}, {self.pool.queue[indices].cell.composition}) from population pool with value {self.pool.queue[indices].value}')
                                        self.ga_logger(self.pool.queue[indices],remove = True,pop_pool = True)
                                        del self.pool.queue[indices]
                                        print([f'Organism ({o.id},{o.cell.composition}): {o.value}' for o in self.pool.queue])

                    else:
                        pass
                else:
                    pass

    def log_memory_usage(self,note=""):
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 **2)
        logging.info(f"[MEMORY] {note} - RSS: {mem:.2f} MB")

    def timeout_handler(self, signum, frame):
        raise TimeoutError(f"Convergence failure on organism, calculation took too long")

    def ga_logger(self, organism, add = False, remove = False, init_pop = False, pop_pool = False):
        file_exists = os.path.exists(os.path.join(os.getcwd(),'ga_history'))
        queue = [f'Organism (ID:{o.id}, Comp:{o.cell.composition}): {o.value}' for o in self.pool.queue]
        promotion = [f'Organism (ID:{o.id}, Comp:{o.cell.composition})' for o in self.pool.promotion_set]
        with open(os.path.join(os.getcwd(),'ga_history'),'a') as f:
            if not file_exists:
                f.write("=" * 128 + "\n")
                f.write("----- Search Space -----" +"\n")
                f.write(f"composition space: {self.composition_space.endpoints}" + "\n")

            if add:
                if init_pop:
                    f.write("-" * 128 + "\n")
                    f.write(f"Adding organism {organism.id} with comp: {organism.cell.composition} to initial population" + "\n")
                if pop_pool:
                    f.write("-" * 128 + "\n")
                    f.write(f"Adding organism {organism.id} with comp: {organism.cell.composition} to pool" + "\n")
                    f.write(f"----- Promotion Set -----" + "\n")
                    f.write("\n".join(promotion) + "\n")
                    f.write(f"-----  Queue -----" + "\n")
                    f.write("\n".join(queue) + "\n")
            if remove:
                if init_pop:
                    f.write("-" * 128 + "\n")
                    f.write(f"Removing organism {organism.id} with comp: {organism.cell.composition} from initial population" + "\n")
                if pop_pool:
                    f.write("-" * 128 + "\n")
                    f.write(f"Removing organism {organism.id} with comp: {organism.cell.composition} from pool" + "\n")
                    f.write(f"----- Promotion Set -----" + "\n")
                    f.write("\n".join(promotion) + "\n")
                    f.write(f"----- Queue -----" + "\n")
                    f.write("\n".join(queue) + "\n")

def main():
    smart_ga = DEEP_GASP()
    pool = smart_ga.initial_population()
    smart_ga.perform_variations(pool)
    print('done!')
if __name__ == "__main__":
    main()
