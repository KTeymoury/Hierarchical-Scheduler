import random

import networkx as nx
from yafs.application import Application, Message, fractional_selectivity
from yafs.core import Sim
from yafs.distribution import deterministicDistributionStartPoint
from yafs.topology import Topology
from yafs.selection import Selection

from utils.placements import MappedPlacement
from utils.populations import MappedPopulation


def dag_to_yafs_application(dag: nx.DiGraph) -> Application:
    '''
    Creates YAFS Application objects based on networkx DAGs.
    '''
    app = Application(name=dag.name)
    modules = []
    messages = {}

    # Handling messages
    for *_, edge in dag.edges(data=True):
        messages[edge['name']] = Message(edge['name'], edge['s'], edge['d'], instructions=edge['instructions'], bytes=edge['bytes'])
        if edge['s'] == 'Source_0':
            app.add_source_messages(messages[edge['name']])
    
    for id, attributes in dag.nodes(data=True):
        # Modules
        if id:
            modules.append({attributes['name']: {'Type': attributes['Type'], 'RAM':attributes['RAM'], 'storage':attributes['storage'],
                                                 'hint': attributes['id'] if 'id' in attributes else id}})
        else: # Sources are special :)
            modules.append({attributes['name']: {'Type': attributes['Type'], 'hint': attributes['id'] if 'id' in attributes else id}})
        
        # Transmissions
        if dag.out_degree(id) > 0: # Normal modules have both message_in and message_out parameter
            for *_, message_in in dag.in_edges(id, data='name'):
                for *_, message_out in dag.out_edges(id, data='name'):
                    app.add_service_module(attributes['name'], messages[message_in], messages[message_out], fractional_selectivity, threshold=1.0)
        else: # Sinks only have message_in
            for *_, message_in in dag.in_edges(id, data='name'):
                app.add_service_module(attributes['name'], messages[message_in])
    
    app.set_modules(modules)

    return app


def network_to_yafs_topology(graph: nx.Graph) -> Topology:
    '''
    Creates a YAFS Topology using a networkx network graph.
    '''
    if not isinstance(graph, nx.classes.graph.Graph):
        raise TypeError

    topology = Topology()
    topology.G = graph
    for id, attributes in graph.nodes(data=True):
        topology.nodeAttributes[str(id)] = attributes.copy()
        topology.nodeAttributes[str(id)]['uptime'] = (0, None)

    topology.__idNode = len(graph.nodes)

    return topology


def create_random_population(dags:nx.DiGraph, network:nx.Graph, number_of_messages=1, seed=None, interval=1000) -> MappedPopulation:
    '''
    Generate a random YAFS Population for testing purposes.

    Note: This function is mostly deprecated and only kept here for future reference.
    '''
    # source_info = (device_id, count, distribution)
    rng = random.Random(seed) if seed else random.Random()
    device_count = network.number_of_nodes() - 1
    result = {}
    for dag in dags:
        sinks = []
        result[dag.name] = ((rng.randint(0, device_count),
                             number_of_messages,
                             deterministicDistributionStartPoint(name="Deterministic", time=interval, start=0)),
                             sinks)
        # Manage sinks later
        '''
        for id, attributes in dag.nodes(data=True):
            if attributes['Type'] == 'SINK':
                sinks.append((device_id, attributes['name']))
        '''
    return MappedPopulation(population_map=result, name='MappedPopulation')


def create_random_placement(dags:nx.DiGraph, network:nx.Graph, number_of_instances=1, seed=None) -> MappedPlacement:
    '''
    Generate a random YAFS Placement for testing purposes.

    Note: This function is mostly deprecated and only kept here for future reference.
    '''
    result = {}
    rng = random.Random(seed) if seed else random.Random()
    device_count = network.number_of_nodes() - 1
    for dag in dags:
        placement = []
        for id, attributes in dag.nodes(data=True):
            if attributes['Type'] == 'MODULE':
                for i in range(number_of_instances):
                    placement.append((attributes['name'], rng.randint(0, device_count)))
        result[dag.name] = placement
    return MappedPlacement(result, name='MappedPlacement')


def create_yafs_simulation(topology:Topology, apps:list[Application], placement:MappedPlacement, population:MappedPopulation, path_selector:Selection, leaker=None):
    '''
    Creates a YAFS Simulation for testing purposes.

    Important Note: This function is deprecated and should NOT be used, it is kept here for future reference.
    '''
    # if issubclass(type(leaker), Metrics):
    simulation = Sim(topology=topology, default_results_path='./sim_trace', metrics=leaker)
    for app in apps:
        simulation.deploy_app2(app, placement, population, path_selector)
    return simulation

