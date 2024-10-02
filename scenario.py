import json
import random
import time
import warnings
from operator import itemgetter
from pathlib import Path

import networkx as nx
import numpy as np
from yafs.application import Application
from yafs.core import Sim
from yafs.distribution import Distribution, deterministicDistributionStartPoint
from yafs.topology import Topology

from assignment import greedy_placement, ilp_placement, ilp_placement_source_included
from utils import graph_generators
from utils.leakers import RunTimeLeaker, RunTimeLeakerWithOutputs
from utils.placements import MappedPlacement
from utils.populations import MappedPopulation, MappedPopulationImproved
from utils import selections
from utils.yafs_tools import dag_to_yafs_application, network_to_yafs_topology

# Partitioning parameters:
LOUVAIN_SEED = 6
RESOLUTION = 1.5
LOUVAIN_THRESHOLD = 1
# Base distance from gateways to cloud:
CLOUD_LATENCY = 60
# Miscellaneous settings:
MAKE_COMPLETE_PATH_CACHE = False
FORCE_COMMONS_FOR_SELECTORS = True
SHARE_PATH_CACHE = True


class Scenario():
    '''
    A world view class in charge of handling practically everything... Scenario objects can be used to configure
    Gymnasium environments and create yafs simulations.
    '''
    SUPPORTED_ASSIGNMENT_METHODS = {'greedy':0, 'ilp':1, 'ilp_improved':2}

    def __init__(self, topology: nx.Graph=None, apps: tuple[nx.DiGraph]=None, initialize=False) -> None:
        '''
        Creates a new scenario using given network topology and application DAGs.
        '''
        self.app_count: int = None
        self.partition_count: int = None
        self.network: nx.Graph = None
        self.apps: tuple[nx.DiGraph] = None
        self.device_info: dict[int: dict] = None
        self.partition_info: tuple[dict] = None
        self.app_info: dict[int: dict] = None
        self.flat_app_info: tuple[tuple[str, int, int, tuple[str, int, int]]] = None
        self.gateways: tuple[str] = None
        self.env_info: dict = None
        self.has_gateways = False
        self.is_initialized: bool = False
        self.is_initializable: bool = False
        self.max_penalty: int = -9999.9

        # Simulation helpers
        self.ready_to_sim = False
        self.sim_apps: tuple[Application] = None
        self.sim_topology: Topology = None
        self.sim_population: MappedPopulation = None
        self.sim_selector: selections.Selection = None
        self.path_cache = {}
        self.population_mapping: dict[str: tuple[tuple[int|str, int, Distribution], tuple[int, str]]] = None

        if topology and apps:
            self.network = topology
            self.apps = apps
            self.is_initializable = True
            if initialize:
                self.update_info()


    def __str__(self):
        if not self.is_initializable:
            return 'Empty Scenario'
        result = 'Initialized Scenario\n' if self.is_initialized else 'Basic Scenario\n'
        result += f'Network Size = {self.network.number_of_nodes()}\nApp Count = {self.app_count}'
        for app in self.apps:
            result += f'\n\tApplication {app.name} with {app.number_of_nodes()-1} services'
        if self.is_initialized:
            result += '\n---------------------------------------------\n'
            result += f'Partition count = {self.partition_count}'
            for i in range(self.partition_count):
                devices = self.partition_info[i]['DEVICES']
                center = self.partition_info[i]['CENTER']
                gateway = self.partition_info[i]['GATEWAY']
                result += f'\n\tPartition {i}: {devices}\n\t\tCenter: {center} | Gateway: {gateway}'
        result += '\n---------------------------------------------\n'
        if self.population_mapping is None:
            result += 'Population is not defined'
        else:
            result += 'Population is defined:'
            for app_name in self.population_mapping:
                gateway = self.population_mapping[app_name][0][0]
                result += f'\n\tApplication {app_name} originates from {gateway}'

        result += '\n\n+ Simulation state: Ready to sim!' if self.ready_to_sim else '\n\n+ Simulation state: Not ready.'
        result += '\n=============================================\n'

        return result

    @staticmethod
    def load_all(directory):
        '''Fully loads an exported scenario.'''
        tmp = Scenario()
        tmp.import_all(directory)
        return tmp
    
    @staticmethod
    def load_basics(directory):
        '''Loads basic information of an exported scenario.'''
        path = Path(directory)

        # Network file
        network_path = path.joinpath('network_topology.gexf')
        if network_path.is_file():
            network_path = network_path.absolute()
        else:
            raise Exception(f'No network graph fount at: {network_path.absolute()}')
        # Application files
        counter = 0
        app_paths = []
        while path.joinpath(f'application_{counter}.gexf').is_file():
            app_paths.append(path.joinpath(f'application_{counter}.gexf').absolute())
            counter += 1
        
        if counter == 0:
            raise Exception(f'No application graph found in directory: {path.absolute()}')
        
        tmp = Scenario()
        tmp.import_basics(network_path, app_paths, False)
        return tmp
    

    def import_basics(self, network_path:str, app_paths:list[str]=None, initialize=True) -> None:
        '''Import basic information from a directory'''
        self.app_count = len(app_paths)
        self.network = nx.read_gexf(network_path)
        nx.relabel_nodes(self.network, lambda x: x if x.startswith('GW') else int(x), copy=False)
        self.apps = tuple([nx.read_gexf(path) for path in app_paths])
        for app in self.apps:
            nx.relabel_nodes(app, lambda x: int(x), copy=False)
        self.is_initializable = True

        if initialize:
            self.update_info()


    def export_basics(self, directory:str, warn_on_gateways=True):
        '''Save basic information to a directory'''
        if not self.is_initializable:
            raise Exception('We don\'t have the basics boss man!')
        if self.gateways and warn_on_gateways:
            warnings.warn('This scenario already has gateways in place. You should either export before adding gateways or use export_all to avoid future problems.')
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        nx.write_gexf(self.network, path.joinpath('network_topology.gexf'))
        counter = 0
        for dag in self.apps:
            nx.write_gexf(dag, path.joinpath(f'application_{counter}.gexf'))
            counter += 1


    def import_all(self, directory: str):
        '''Import a complete scenario from a directory'''
        path = Path(directory)

        network_path = path.joinpath('network_topology.gexf')
        if network_path.is_file():
            network_path = network_path.absolute()
        else:
            raise Exception(f'No network graph with following address found: {network_path.absolute()}')
        
        counter = 0
        app_paths = []
        while path.joinpath(f'application_{counter}.gexf').is_file():
            app_paths.append(path.joinpath(f'application_{counter}.gexf').absolute())
            counter += 1
        
        if counter == 0:
            raise Exception(f'No application graph found in directory: {path.absolute()}')
        
        info_path = path.joinpath('scenario_information.json')
        if info_path.is_file():
            info_path = info_path.absolute()
        else:
            raise Exception(f'No information json was found at: {info_path.absolute()}\nHint: You might want to use import_basic')
        
        self.import_basics(network_path, app_paths, False)
        
        jsonified = ''
        with open(info_path, 'r') as f:
            jsonified = json.loads(f.read())
        
        self.partition_count = jsonified['partition_count']
        
        self.app_count = jsonified['app_count']
        
        self.device_info = {int(k): jsonified['device_info'][k] for k in jsonified['device_info']}
        
        self.partition_info = jsonified['partition_info']
        gateways = []
        for partition in self.partition_info:
            partition['SUBGRAPH'] = self.network.subgraph(partition['SUBGRAPH'])
            if partition['GATEWAY']:
                self.has_gateways = True
                gateways.append(partition['GATEWAY'])
        if self.has_gateways:
            self.gateways = tuple(gateways)
        self.partition_info = tuple(self.partition_info)
        
        self.flat_app_info = jsonified['flat_app_info']

        self.app_info = {int(k): jsonified['app_info'][k] for k in jsonified['app_info']}
        lookup = {dag.name: dag for dag in self.apps}
        for k in self.app_info:
            modules = tuple([tuple(module_info) for module_info in self.app_info[k]['MODULES']])
            self.app_info[k]['MODULES'] = modules
            self.flat_app_info[k][3] = modules
            self.app_info[k]['DAG'] = lookup[self.app_info[k]['DAG']]

        self.flat_app_info = tuple(self.flat_app_info)

        if jsonified['population_mapping']:
            self.population_mapping = jsonified['population_mapping']
            for item in self.population_mapping:
                source, sink = self.population_mapping[item]
                # Create distribution object
                if source[2]['type'] == 'deterministicDistributionStartPoint':
                    source[2] = deterministicDistributionStartPoint(name=source[2]['name'], time=source[2]['time'], start=source[2]['start'])
                else: # TODO: handle other types if needed
                    source[2] = None
                self.population_mapping[item] = (tuple(source), tuple(sink))

        # create gym environment config
        partition_info = self.partition_info
        app_info = self.app_info
        self.env_info = {'task_count': self.app_count,
                         'partition_count': self.partition_count,
                         'partition_info':np.array([(partition['TOTAL_IPT'], partition['TOTAL_RAM']) for partition in partition_info], dtype=np.int64),
                         'task_info':np.array([(app_info[i]['TOTAL_INSTRUCTION'], app_info[i]['TOTAL_RAM'], i) for i in app_info] + [(0, 0, self.app_count)], dtype=np.int64),
                         'violation_penalty': self.max_penalty,
                         'parent_scenario': self}
        
        self.is_initialized = True


    def export_all(self, directory:str):
        '''Save the entire scenario to a directory'''
        if not self.is_initialized:
            raise Exception('Scenario is not initialized!')
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.export_basics(directory, False)
        jsonified = json.dumps({'partition_count': self.partition_count,
                                'app_count': self.app_count,
                                'device_info': self.device_info,
                                'partition_info': self.partition_info,
                                'app_info': self.app_info,
                                'flat_app_info': self.flat_app_info,
                                'population_mapping': self.population_mapping},
                                indent=2, cls=CustomEncoder)
        with open(path.joinpath('scenario_information.json').absolute(), 'w') as f:
            f.write(jsonified)


    def get_obvious_clouds(self):
        results = []
        for device in self.device_info:
            if 'cloud' in self.device_info[device]:
                results.append(device)
        return results
    

    def update_info(self):
        '''
        Given a fresh scenario, this function does a number of things:
            1. Extracts device informations from network graph.
            2. Partitions the network graph into communities and selects a center for each partition.
            3. Calculates total resource for every partition
            4. Extracts application informations from application DAGs and calculates total required resource by them.
            5. Creates an environment config dictionary for Gymnasium environment.
            6. Elevates the state of this scenario to initiated.
        '''
        if not self.is_initializable:
            raise Exception('Scenario not initializable!')
        
        self.app_count = len(self.apps)
        # set device_info
        device_info = dict(self.network.nodes(data=True))
        self.device_info = device_info
        # set partition_info
        partition_info = []
        comms = nx.community.louvain_communities(self.network, weight='speed_measure', seed=LOUVAIN_SEED, resolution=RESOLUTION, threshold=LOUVAIN_THRESHOLD)
        self.partition_count = len(comms)
        for partition in comms:
            current_ram, current_storage, current_ipt = 0, 0, 0
            for id in partition:
                resource = device_info[id]
                current_ram += resource['RAM']
                current_storage += resource['storage']
                current_ipt += resource['IPT']
            subg = self.network.subgraph(partition)
            center = sorted(nx.closeness_centrality(subg, distance='latency_measure').items(), key=itemgetter(1), reverse=True)[0][0]
            partition_info.append({'TOTAL_RAM': current_ram,
                                   'TOTAL_STORAGE': current_storage,
                                   'TOTAL_IPT': current_ipt,
                                   'DEVICES': sorted(partition, reverse=True, key=lambda x:device_info[x]['IPT']),
                                   'CENTER': center,
                                   'GATEWAY': None,
                                   'SUBGRAPH': subg})
        self.partition_info = tuple(partition_info)

        # get application info
        flat_app_info = []
        app_info = {}
        for id in range(self.app_count):
            # app_info
            current_dag = self.apps[id]
            current_ram, current_storage, current_instruction = 0, 0, 0
            # flat_app_info
            current_modules = []
            # mixed
            for i, info in current_dag.nodes(data=True):
                if i:
                    # app_info
                    current_ram += info['RAM']
                    current_storage += info['storage']
                    # flat_app_info
                    *_, tmp_instructions = current_dag.in_edges(i, data='instructions')
                    current_modules.append((info['name'], sum(tmp_instructions), info['RAM']))
            # app_info
            for *_, ins in current_dag.edges(data='instructions'):
                current_instruction += ins
            current_modules = tuple(current_modules)
            app_info[id] = {'TOTAL_RAM': current_ram,
                            'TOTAL_STORAGE': current_storage,
                            'TOTAL_INSTRUCTION': current_instruction,
                            'NAME': current_dag.name,
                            'DAG': current_dag,
                            'MODULES': current_modules}
            flat_app_info.append((current_dag.name, current_instruction, current_ram, current_modules))
        
        self.app_info = app_info
        self.flat_app_info = flat_app_info
        # create gym environment config
        self.env_info = {'task_count': self.app_count,
                         'partition_count': self.partition_count,
                         'partition_info':np.array([(partition['TOTAL_IPT'], partition['TOTAL_RAM']) for partition in partition_info], dtype=np.int64),
                         'task_info':np.array([(app_info[i]['TOTAL_INSTRUCTION'], app_info[i]['TOTAL_RAM'], i) for i in app_info] + [(0, 0, self.app_count)], dtype=np.int64),
                         'violation_penalty': self.max_penalty,
                         'parent_scenario': self}
        
        print('Info: Successfully initiated scenario.')
        self.is_initialized = True

    
    def add_gateways(self):
        '''Adds external gateway near every partition to emulate users.'''
        gw_list = self.gateways

        # Check if gateways already exist in scenario if they don't try and extract from partition info
        if not gw_list:
            gw_list = tuple([info['GATEWAY'] for info in self.partition_info if info['GATEWAY'] is not None])
            self.gateways = gw_list

        # If there isn't at least one gateway in partition info create new gateways
        if len(gw_list) == 0:
            cloud_ids = []
            gw_list = []
            centers = []
            for index in range(self.partition_count):
                if len(self.partition_info[index]['DEVICES']) == 1:
                    if 'cloud' in self.device_info[self.partition_info[index]['DEVICES'][0]]:
                        cloud_ids.append(self.partition_info[index]['DEVICES'][0])
                        self.partition_info[index]['GATEWAY'] = None
                        gw_list.append(None)
                        print(f'Found a cloud node with device ID {cloud_ids[-1]}')
                        continue
                    else:
                        warnings.warn(f'Encountered a partition of 1 device that is not cloud at partition {index}.')
                
                self.partition_info[index]['GATEWAY'] = f'GW{index}'
                gw_list.append(f'GW{index}')
                centers.append(self.partition_info[index]['CENTER'])
            
            self.gateways = tuple([gw for gw in gw_list if not gw is None])
            
            node_info = []
            edge_info = []
            for i in range(len(gw_list)):
                if gw_list[i] == None:
                    print(f'Skipping partition {i} due to no gateway policy for cloud nodes.')
                    continue
                node_info.append((gw_list[i], {'IPT': 10, 'RAM': 10**6, 'storage': 10**6}))
                for cloud_id in cloud_ids:
                    pr = CLOUD_LATENCY + 2
                    edge_info.append((gw_list[i], cloud_id, {'BW': 1000, 'PR': pr, 'latency_measure': 1+pr, 'speed_measure': 1/(1+pr)}))
                for j in range(len(centers)):
                    pr = abs(j-i) * 10 + 2
                    edge_info.append((gw_list[i], centers[j], {'BW': 1000, 'PR': pr, 'latency_measure': 1+pr, 'speed_measure': 1/(1+pr)}))
            self.network.add_nodes_from(node_info)
            self.network.add_edges_from(edge_info)
        
        print('Info: Successfully added gateways.')
        self.has_gateways = True
        
    
    def remove_gateways(self):
        '''Removes gateways from this scenario.'''
        print('Attempting to remove following nodes from graph:')
        print(f'\t{self.gateways}')
        print('WARNING: Population information wil also be reset.')

        self.network.remove_nodes_from(self.gateways)
        self.has_gateways = False
        self.gateways = None
        for partition in self.partition_info:
            partition['GATEWAY'] = None
        self.population_mapping = None
        self.sim_population = None
        print('Info: Successfully removed gateways.')
    

    def initialize_population(self, population_map= None, seed=None, number_of_messages=1, message_interval=5000, use_improved=True):
        '''
        Creates a YAFS compatible population object based on 'population_map' if provided, otherwise creates a random
        population for testing purposes.
        '''
        if self.population_mapping is None:
            if population_map is None:
                rng = random.Random(seed) if seed else random.Random()
                tmp = {}
                self.population_mapping = tmp
                sinks = tuple()
                for dag in self.apps:
                    tmp[dag.name] = ((rng.choice(self.gateways),
                                    number_of_messages,
                                    deterministicDistributionStartPoint(name="Deterministic", time=message_interval, start=0)),
                                    sinks)
            else: 
                self.population_mapping = population_map

        if use_improved:
            self.sim_population = MappedPopulationImproved(self.population_mapping, name='AutoGeneratedPopulation')
        else:
            self.sim_population = MappedPopulation(self.population_mapping, name='AutoGeneratedPopulation')
        print('Info: Successfully initialized population.')
    

    def _cache_paths(self):
        '''Internal function to initiate a scenario wide path cache for network.'''
        self.path_cache = {}
        for partition in self.partition_info:
            subg = partition['SUBGRAPH']
            paths = nx.all_pairs_dijkstra_path(subg, weight='latency_measure')
            for source, path_to in paths:
                for destination in path_to:
                    if source == destination:
                        continue
                    self.path_cache[source, destination] = (tuple(path_to[destination]),)


    def prepare_simulation(self):
        '''Creates YAFS components and elevates this scenario to a simulation-ready state.'''
        # See if it's ready to simulate already
        if self.ready_to_sim:
            return False
        
        # Asking an uninitialized scenario to make simulation is bad
        if not self.is_initialized:
            warnings.warn('!!!IMPORTANT!!!\n\
                          Asking to prepare a simulation without even initializing this scenario are we?\n\
                          Initializing this scenario for you but this is probably not what you meant to do!!!')
            self.update_info()

        # See if there are gateways in current network and generate if there aren't
        if not self.has_gateways:
            self.add_gateways()
        
        # Create a random population if there isn't a population in current scenario
        if self.sim_population is None:
            self.initialize_population()
        
        # Create YAFS Topology from network graph 
        self.sim_topology = network_to_yafs_topology(self.network)

        # Create YAFS Applications from DAGs
        self.sim_apps = tuple([dag_to_yafs_application(app) for app in self.apps])

        # Yafs selector creation goes here
        ## See if paths need to be cached for faster lookup
        if MAKE_COMPLETE_PATH_CACHE:
            self._cache_paths()
        
        if FORCE_COMMONS_FOR_SELECTORS:
            selections.USE_COMMONS = True
            if SHARE_PATH_CACHE:
                selections._initialize_commons(self.path_cache, self.partition_info, self.gateways)
            else:
                selections._initialize_commons({}, self.partition_info, self.gateways)
        else:
            selections.USE_COMMONS = False
        
        if SHARE_PATH_CACHE:
            #self.sim_selector = selections.CachedSelector(self.path_cache)
            self.sim_selector = selections.PartitionedSelector(self.gateways, self.partition_info, self.path_cache)
        else:
            #self.sim_selector = selections.CachedSelector()
            self.sim_selector = selections.PartitionedSelector(self.gateways, self.partition_info)

        # Set the flag
        self.ready_to_sim = True
        print('Info: Ready to simulate!')

        return True


    def generate_service_assignment_map(self, partition_assignment: np.ndarray, method:str='greedy', time_assignment=False):
        '''
        Creates an assignment dictionary based on the provided app to partition map using either 'greedy' or 'ilp' methods.

        Note: If you need to profile assignment process set the 'time_assignment' flag, doing so will make this function 
        return a list of wall clock times along with actual results.
        '''
        if not method in Scenario.SUPPORTED_ASSIGNMENT_METHODS:
            raise Exception(f'specified method "{method}" is not supported please use one of the following: {Scenario.SUPPORTED_ASSIGNMENT_METHODS.keys()}')
        
        method = Scenario.SUPPORTED_ASSIGNMENT_METHODS[method]
        start_time = 0
        time_list = []
        global_placement_map = {app_name: [] for app_name, *_ in self.flat_app_info}

        for i in range(self.partition_count):
            # List IDs of applications assigned to current partition.
            selected_apps = partition_assignment[:,i].nonzero()[0]
            if len(selected_apps) == 0:
                warnings.warn(f'Nothing was assigned to partition {i}')
                continue

            # Time START_TIME if necessary
            if time_assignment:
                start_time = time.time()
            
            # Do the thing based on selected method :)
            if method == 0: # Greedy
                sub_list = [self.flat_app_info[index] for index in selected_apps]
                # Following is more efficient but weird behaviour in single item selection
                #sub_list = itemgetter(*selected_apps)(self.flat_app_info)
                success, nice, partition_placement = greedy_placement(self.partition_info[i]['SUBGRAPH'], sub_list)
                for app in partition_placement:
                    global_placement_map[app].extend(partition_placement[app])
                if success:
                    if nice:
                        print(f'INFO: Greedy assignment found a nice placement solution for partition {i}.')
                    else:
                        print(f'INFO: Greedy assignment could not find a nice solution for partition {i}.')
                else:
                    warnings.warn(f'IMPORTANT: Service placement over partition {i} was NOT successful.\
                                  The resulting dictionary will most likely be incomplete.')
            
            elif method == 1: # ILP
                sub_list = [self.apps[index] for index in selected_apps]
                success, optimal, partition_placement = ilp_placement(self.partition_info[i]['SUBGRAPH'], sub_list)
                for app in partition_placement:
                    global_placement_map[app].extend(partition_placement[app])
                if success:
                    if optimal:
                        print(f'INFO: ILP assignment found an optimal placement solution for partition {i}.')
                    else:
                        print(f'INFO: ILP assignment could not find an optimal solution for partition {i}.')
                else:
                    warnings.warn(f'IMPORTANT: Service placement over partition {i} was NOT successful.\
                                  The resulting dictionary will most likely be incomplete.')
            
            elif method == 2: # ILP IMPROVED (Uses partition centers as source location)
                sub_list = [self.apps[index] for index in selected_apps]
                success, optimal, partition_placement = ilp_placement_source_included(self.partition_info[i]['SUBGRAPH'], self.partition_info[i]['CENTER'], sub_list)
                for app in partition_placement:
                    global_placement_map[app].extend(partition_placement[app])
                if success:
                    if optimal:
                        print(f'INFO: ILP assignment found an optimal placement solution for partition {i}.')
                    else:
                        print(f'INFO: ILP assignment could not find an optimal solution for partition {i}.')
                else:
                    warnings.warn(f'IMPORTANT: Service placement over partition {i} was NOT successful.\
                                  The resulting dictionary will most likely be incomplete.')
            
            else: # UNKNOWN...
                raise NotImplementedError('HOW ON EARTH DID THIS HAPPEN?!!')
            
            # Time END_TIME if necessary
            if time_assignment:
                tmp_time = time.time() - start_time
                print(f'This step took {tmp_time:9.6f} seconds')
                time_list.append(tmp_time)
                start_time = time.time()
        
        if time_assignment:
            return global_placement_map, time_list
        
        return global_placement_map


    def test_partition_assignment(self, partition_assignment: np.ndarray, dedicated_selector=False) -> float:
        '''
        Using the greedy_placement function, places the application over physical devices based on provided assignment map,
        then, using the generated placement map, runs a simulation in YAFS and returns the negated average run time of
        applications in order to be used as reward/penalty in RL algorithm.

        Note: You can use this function as a reference for 'how to get greedy assignment to work'. :)
        '''
        if not self.ready_to_sim:
            raise Exception('Please prepare simulation infos first.')
        if FORCE_COMMONS_FOR_SELECTORS:
            if not selections._common_set:
                if SHARE_PATH_CACHE:
                    selections._initialize_commons(self.path_cache, self.partition_info, self.gateways)
                else:
                    selections._initialize_commons({}, self.partition_info, self.gateways)
        ### PL_START = time.time()
        # Create MappedPlacement
        global_placement_map = {app_name: [] for app_name, *_ in self.flat_app_info}
        for i in range(self.partition_count):
            selected_apps = partition_assignment[:,i].nonzero()[0]
            if len(selected_apps) == 0:
                warnings.warn(f'Nothing was placed on {i}')
                continue
            sub_list = [self.flat_app_info[index] for index in selected_apps]
            # more efficient but weird behaviour in single item selection
            # sub_list = itemgetter(*selected_apps)(self.flat_app_info)
            success, nice, partition_placement = greedy_placement(self.partition_info[i]['SUBGRAPH'], sub_list)
            if success:
                for app in partition_placement:
                    global_placement_map[app].extend(partition_placement[app])
            else:
                return self.max_penalty
        sim_placement = MappedPlacement(global_placement_map, name='MappedPlacement')
        ### PL_END = time.time()

        ### SIM_START = time.time()
        sim_selector = self.sim_selector
        if dedicated_selector:
            sim_selector = selections.RoundRobinPartitionedSelector(self.gateways,self.partition_info, self.path_cache)
        # Create simulation with a metrics leaker
        #simulation = Sim(topology=self.sim_topology, default_results_path='./sim_trace')
        simulation = Sim(topology=self.sim_topology, default_results_path='./sim_trace', metrics=RunTimeLeaker())
        for app in self.sim_apps:
            simulation.deploy_app2(app, sim_placement, self.sim_population, sim_selector)
        
        # These are the minimum, maximum, average app execution time in current setup (OOPS changed this)
        average = simulation.run(10000, silent=True)
        ### SIM_END = time.time()
        ### print(f'Placement took: {PL_END-PL_START}\nSim took: {SIM_END-SIM_START}')
        # Negate and return the average for agent
        return -average
    

    def simulate_placement(self, placement_map:dict, output_dir:str=None, simulation_time=10000, selector=selections.PartitionedSelector):
        '''
        Runs a YAFS simulation using the given placement map. 'placement_map' should be compatible with MappedPlacement.
        By default this function returns the average runtime of applications and dumps results inside specified output
        directory.
        '''
        if not self.ready_to_sim:
            raise Exception('Please prepare simulation informations first.')
        
        path = None
        if output_dir:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
        
        sim_placement = MappedPlacement(placement_map, name='MappedPlacement')

        sim_selector = None
        if selector is selections.CachedSelector:
            sim_selector = selector(self.path_cache)
        else:
            sim_selector = selector(self.gateways, self.partition_info, self.path_cache)
        
        simulation = None
        if output_dir:
            simulation = Sim(topology=self.sim_topology, metrics=RunTimeLeakerWithOutputs(str(path.joinpath('sim_trace'))))
        else:
            simulation = Sim(topology=self.sim_topology, metrics=RunTimeLeaker())

        for app in self.sim_apps:
            simulation.deploy_app2(app, sim_placement, self.sim_population, sim_selector)
        
        average = simulation.run(simulation_time, silent=True)
        return average


    def evaluate_ilp_assignment(self, partition_assignment: np.ndarray, dedicated_selector=False) -> float:
        '''
        Uses ILP assignment to place applications over physical devices inside designated partitions and evaluates the
        placement using a YAFS simulation. This function returns the average run time of applications.

        Note: You can use this function as a reference for 'how to get ILP assignment to work'. :)
        '''
        if not self.ready_to_sim:
            raise Exception('Please prepare simulation informations first.')
        
        ### PL_START = time.time()
        # Create MappedPlacement
        global_placement_map = {app_name: [] for app_name, *_ in self.flat_app_info}
        for i in range(self.partition_count):
            selected_apps = partition_assignment[:,i].nonzero()[0]
            if len(selected_apps) == 0:
                warnings.warn(f'Nothing was placed on {i}')
                continue
            sub_list = [self.apps[index] for index in selected_apps]
            # more efficient but weird behaviour in single item selection
            # sub_list = itemgetter(*selected_apps)(self.flat_app_info)
            success, nice, partition_placement = ilp_placement(self.partition_info[i]['SUBGRAPH'], sub_list)
            if success:
                for app in partition_placement:
                    global_placement_map[app].extend(partition_placement[app])
            else:
                return self.max_penalty
        sim_placement = MappedPlacement(global_placement_map, name='MappedPlacement')
        ### PL_END = time.time()

        ### SIM_START = time.time()
        sim_selector = self.sim_selector
        if dedicated_selector:
            sim_selector = selections.RoundRobinPartitionedSelector(self.gateways,self.partition_info, self.path_cache)
        # Create simulation with a metrics leaker
        #simulation = Sim(topology=self.sim_topology, default_results_path='./sim_trace')
        simulation = Sim(topology=self.sim_topology, default_results_path='./sim_trace', metrics=RunTimeLeaker())
        for app in self.sim_apps:
            simulation.deploy_app2(app, sim_placement, self.sim_population, sim_selector)
        
        # These are the minimum, maximum, average app execution time in current setup (OOPS changed this)
        average = simulation.run(10000, silent=True)
        ### SIM_END = time.time()
        ### print(f'Placement took: {PL_END-PL_START}\nSim took: {SIM_END-SIM_START}')
        # Negate and return the average for agent
        return -average

    
    def evaluate_json_placement(self, path: str) -> float:
        '''
        Runs a YAFS simulation based on a placement map provided in a json file. Generally used as a compatibility function
        for default YAFS stuff.

        Note: You can use this function as-is or as a reference to evaluate placement.json files created by other algorithms
        or YAFS's default placement schemes.
        '''
        if not self.ready_to_sim:
            raise Exception('Please prepare simulation infos first.')
        placement_map = json.load(open(path))
        ### PL_START = time.time()
        # Create MappedPlacement
        global_placement_map = {app_name: [] for app_name, *_ in self.flat_app_info}

        for item in placement_map['initialAllocation']:
            if item['module_name'] == 'Source_0':
                continue
            global_placement_map[item['app']].append((item['module_name'], item['id_resource']))
        sim_placement = MappedPlacement(global_placement_map, name='MappedPlacement')
        ### PL_END = time.time()

        ### SIM_START = time.time()
        # Create simulation with a metrics leaker
        simulation = Sim(topology=self.sim_topology, default_results_path='./sim_trace', metrics=RunTimeLeaker())
        selector = selections.CachedSelector()
        for app in self.sim_apps:
            simulation.deploy_app2(app, sim_placement, self.sim_population, selector)
        
        # These are the minimum, maximum, average app execution time in current setup (OOPS changed this)
        average = simulation.run(10000, silent=True)
        ### SIM_END = time.time()
        ### print(f'Placement took: {PL_END-PL_START}\nSim took: {SIM_END-SIM_START}')
        # Negate and return the average for agent
        return -average


# pass this as cls
class CustomEncoder(json.JSONEncoder):
    '''
    A custom json encoder to encode various parts of Scenario objects that are not supported by json module.
    '''
    def default(self, obj):
        if isinstance(obj, nx.DiGraph):
            return obj.name
        elif isinstance(obj, nx.Graph):
            return list(obj.nodes())
        elif isinstance(obj, deterministicDistributionStartPoint):
            return {'type': 'deterministicDistributionStartPoint', 'name': obj.name, 'start': obj.start, 'time': obj.time}
        return json.JSONEncoder.default(self, obj)


def create_scenario(app_count=10, approximate_device_count=100)->Scenario:
    '''
    creates a random scenario using graph_tools. Acts as a "how to" for their usage.
    '''
    topology=None
    if approximate_device_count>40:
        mean = approximate_device_count//10
        topology = graph_generators.create_tiny_network(10, (mean-3, mean+3))
    else:
        topology = graph_generators.create_tiny_network(10, (approximate_device_count//10, approximate_device_count//10+1))
    dags = graph_generators.create_app_batch(app_count)
    return Scenario(topology, dags)

def create_random_scenario()->Scenario:
    rng = random.Random(46)

    topology = nx.barabasi_albert_graph(300, 50)

    # Generate and set node attributes
    attributes = {}
    for id in range(300):
        attributes[id] = {'IPT': rng.randint(10, 40) * 100, # Let's go with MHz for unit
                          'RAM': rng.choice((1, 2, 4, 8, 16, 32, 64)) * 1000, # Unit = MB
                          'storage': 500_000} # Unit = MB
    nx.set_node_attributes(topology, attributes)
    
    # Generate and set edge attributes
    edge_attribute = {}
    for source, destination, attrib in topology.edges(data=True):
        attrib['PR'] = rng.randint(0, 2)
        attrib['BW'] = rng.randint(20, 50) * 10
        
        attrib['latency_measure'] = attrib['PR'] + 1000 / attrib['BW']
        attrib['speed_measure'] = 1 / attrib['latency_measure']

    dags = graph_generators.create_app_batch(10)

    return Scenario(topology, dags)
