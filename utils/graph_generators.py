import networkx as nx
from utils import graph_tools
import random


def create_tiny_network(community_count=5, community_node_count:tuple[int, int]=(5,10), has_cloud=True, seed=None) -> nx.Graph:
    '''
    Creates a tiny network with a single cloud component. Given the tiny status it can hold up to 12 community of nodes.
    Community in this function means a set of closely knitted nodes and is not necessarily an actual community detectable
    by graph partitioning algorithm used.
    '''
    if community_count>13:
        raise Exception("Mate that's not tiny... maximum community count is 12.")
    
    rng = random.Random(seed) if seed else random.Random()

    attributes = {}
    id2community = []
    last_id = 0
    for i in range(community_count):
        count = rng.randrange(*community_node_count)
        id2community.extend([i] * count)

        for j in range(last_id, last_id+count):
            attributes[j] = {'IPT': rng.randint(10, 40) * 100, # Let's go with MHz for unit
                             'RAM': rng.choice((1, 2, 4, 8, 16, 32, 64)) * 1000, # Unit = MB
                             'storage': 500_000} # Unit = MB
        
        last_id += count
    
    if has_cloud:
        id2community.append(community_count)

        attributes[last_id] = {'IPT': 400_000, # MHz
                               'RAM': 999_999_999, # MB
                               'storage': 999_999_999, # MB
                               'cloud': True}
        
        last_id += 1
    
    id2community = tuple(id2community)
    g = nx.complete_graph(last_id)
    nx.set_node_attributes(g, attributes)

    for source, destination, attrib in g.edges(data=True):
        source_community = id2community[source]
        destination_community = id2community[destination]

        distance = abs(source_community-destination_community) + 1

        attrib['PR'] = distance * 10 if source_community < community_count and destination_community < community_count else 60
        attrib['BW'] = rng.randint(20, 40) * 10 if source_community < community_count and destination_community < community_count else 1000
        
        attrib['latency_measure'] = attrib['PR'] + 1000 / attrib['BW']
        attrib['speed_measure'] = 1 / attrib['latency_measure']
        
    return g


def create_huge_network(community_count=15, community_node_count:tuple[int, int]=(20, 30), has_cloud=True, seed=None) -> nx.Graph:
    '''
    Creates a huge network of physical nodes with a number of cloudlets distributed evenly among communities. A huge setup consists
    of 13 or more communities and number of cloudlets are dynamically determined by community count (one for every 12 community).
    '''
    if community_count<13:
        raise Exception("For 12 and less communities use tiny one.")
    
    rng = random.Random(seed) if seed else random.Random()

    cloud_count = (community_count + 11) // 12 if has_cloud else 0

    attributes = {}
    id2community = []
    last_id = 0

    for i in range(community_count):
        count = rng.randrange(*community_node_count)
        id2community.extend([i] * count)

        for j in range(last_id, last_id+count):
            attributes[j] = {'IPT': rng.randint(10, 40) * 100, # Let's go with MHz for unit
                             'RAM': rng.choice((1, 2, 4, 8, 16, 32, 64)) * 1000, # Unit = MB
                             'storage': 500_000} # Unit = MB
        
        last_id += count
    
    for i in range(cloud_count):
        id2community.append(community_count+i)

        attributes[last_id] = {'IPT': 500_000//cloud_count, # MHz
                               'RAM': 999_999_999, # MB
                               'storage': 999_999_999, # MB
                               'cloud': True}
        
        last_id += 1
    
    id2community = tuple(id2community)
    g = nx.complete_graph(last_id)
    nx.set_node_attributes(g, attributes)

    for source, destination, attrib in g.edges(data=True):
        source_community = id2community[source]
        destination_community = id2community[destination]

        distance = abs(source_community-destination_community) + 1

        attrib['PR'] = distance * 10 if source_community < community_count and destination_community < community_count else 60
        attrib['BW'] = rng.randint(20, 40) * 10 if source_community < community_count and destination_community < community_count else 1000
        
        attrib['latency_measure'] = attrib['PR'] + 1000 / attrib['BW']
        attrib['speed_measure'] = 1 / attrib['latency_measure']
        
    return g


def create_app_batch(app_count=5, service_count_range:tuple[int, int]=(12,20), seed=None) -> list[nx.DiGraph]:
    '''
    Creates a list of Scenario compatible apps.
    '''
    rng = random.Random(seed) if seed else random.Random()
    apps = []
    for i in range(app_count):
        app = graph_tools.create_random_applicaton_dag(rng.randint(*service_count_range),
                                                       f'Application_{i}',
                                                       instructions_mag=10,
                                                       instructions_range=(5, 40),
                                                       RAM_range=(5, 500, 5),
                                                       storage_range=(10, 1000),
                                                       bytes_range=(100, 500, 20))
        apps.append(app)
    return apps
