import random

import matplotlib.pyplot as plt
import networkx as nx


def create_random_applicaton_dag_with_dummy_sink(node_count, application_name, instructions_range=(1,500), instructions_mag=10**3, RAM_range=(1000,40000), storage_range=(1000,40000), bytes_range=(10,1000), seed=None, graph_seed=None) -> nx.DiGraph:
    '''
    Crates a random application graph with given specs. Applications created through this method have a dummy sink. A dummy sink is
    a node that receives a message after completion of every branch in DAG.

    Note 1: DAGs created by this function have a gateway friendly source node meaning source node omits a single message.
    Note 2: Use this function if you want to simulate feedbacks
    '''
    if node_count < 2:
        raise Exception('DAG needs atleast 2 nodes mate.')
    # Handle RNG seeds
    rng = random.Random(seed) if seed else random.Random()
    # Create a graph with 1 less node then add a single source to DAG
    graph = nx.gn_graph(node_count-1, seed=graph_seed).reverse()
    nx.relabel_nodes(graph, lambda x: x+1, copy=False)
    graph.add_node(0)
    graph.add_edge(0, 1)
    graph.name = application_name

    # Generate and set node attributes
    attributes = {0:{'id': 0, 'name': 'Source_0', 'Type': 'SOURCE'}}
    module_counter = 0
    sink_id = node_count + 1
    # Dummy Sink
    graph.add_node(sink_id, id=sink_id, name='Sink_DUMMY', RAM=0, storage=0, Type='SINK')
    
    for id, degree in graph.out_degree():
        if id == 0 or id == sink_id: # Source Node or Dummy Sink
            continue
        elif degree == 0: # Previous Sink Nodes
            graph.add_edge(id, sink_id)
            
        attributes[id] =   {'id': id,
                            'name': f'Module_{module_counter}',
                            'RAM': rng.randrange(*RAM_range),
                            'storage': rng.randrange(*storage_range),
                            'Type': 'MODULE'}
        module_counter += 1
    nx.set_node_attributes(graph, attributes)
    
    # Generate and set edge attributes
    edge_attribute = {}
    edge_counter = 0
    names = nx.get_node_attributes(graph, "name")
    for edge in graph.edges:
        if edge[1]==sink_id: # CHANGE 'bytes' HERE IF YOU WANT ACTUAL FEEDBACK MESSAGES
            edge_attribute[edge] = {'s': names[edge[0]],
                                    'd': names[edge[1]],
                                    'id': edge_counter,
                                    'name': f'Message_{edge_counter}',
                                    'bytes': 0,
                                    'instructions': 0}
        else:
            edge_attribute[edge] = {'s': names[edge[0]],
                                    'd': names[edge[1]],
                                    'id': edge_counter,
                                    'name': f'Message_{edge_counter}',
                                    'bytes': rng.randrange(*bytes_range),
                                    'instructions': rng.randrange(*instructions_range)*instructions_mag}
        edge_counter += 1
    nx.set_edge_attributes(graph, edge_attribute)

    return graph


def create_random_applicaton_dag(node_count, application_name, sinks_as_modules=True, instructions_range=(1,500), instructions_mag=10**3, RAM_range=(1000,40000), storage_range=(1000,40000), bytes_range=(10,1000), seed=None, graph_seed=None) -> nx.DiGraph:
    '''
    Creates a random application DAG according to given specs. Generally speaking this functions is what you want :).
    
    Note 1: DAGs created by this function have a gateway friendly source node meaning source node omits a single message.
    Note 2: If 'sinks_as_modules' flag is set, sinks would be tagged as 'MODULE' too. Use this if you want YAFS to execute
    sinks too.
    '''
    if node_count < 2:
        raise Exception('DAG needs atleast 2 nodes mate.')
    # Handle RNG seeds
    rng = random.Random(seed) if seed else random.Random()
    # Create a graph with 1 less node then add a single source to DAG
    graph = nx.gn_graph(node_count-1, seed=graph_seed).reverse()
    nx.relabel_nodes(graph, lambda x: x+1, copy=False)
    graph.add_node(0)
    graph.add_edge(0, 1)
    graph.name = application_name

    # Generate and set node attributes
    attributes = {0:{'id': 0, 'name': 'Source_0', 'Type': 'SOURCE'}}
    module_counter = 0
    sink_counter = 0
    for id, degree in graph.out_degree():
        if id == 0: # Source Node
            continue
        elif degree == 0: #Sink Nodes
            attributes[id] = {'id': id,
                            'name': f'Sink_{sink_counter}',
                            'RAM': rng.randrange(*RAM_range),
                            'storage': rng.randrange(*storage_range),
                            'Type': 'MODULE' if sinks_as_modules else 'SINK'}
            sink_counter += 1
        else: #Modules
            attributes[id] = {'id': id,
                              'name': f'Module_{module_counter}',
                              'RAM': rng.randrange(*RAM_range),
                              'storage': rng.randrange(*storage_range),
                              'Type': 'MODULE'}
            module_counter += 1
    nx.set_node_attributes(graph, attributes)
    
    # Generate and set edge attributes
    edge_attribute = {}
    edge_counter = 0
    names = nx.get_node_attributes(graph, "name")
    for edge in graph.edges:
        edge_attribute[edge] = {'s': names[edge[0]],
                                'd': names[edge[1]],
                                'id': edge_counter,
                                'name': f'Message_{edge_counter}',
                                'bytes': rng.randrange(*bytes_range),
                                'instructions': rng.randrange(*instructions_range)*instructions_mag}
        edge_counter += 1
    nx.set_edge_attributes(graph, edge_attribute)

    return graph


def create_random_topology_graph(node_count, minimum_edge, IPT_range=(1,400), IPT_mag=10**6, RAM_range=(10**6,32*10**6), storage_range=(1000,40000), BW_range=(10,1000), PR_range=(1,5), default_packet_size=1000, seed=None, graph_seed=None) -> nx.Graph:
    '''
    Creates a random network topology graph according to given specs.
    
    Note: This is a mostly deprecated function kept as a reference.
    '''
    # Handle RNG seeds
    rng = random.Random(seed) if seed else random.Random()
    graph = nx.barabasi_albert_graph(node_count, minimum_edge, graph_seed)

    # Generate and set node attributes
    attributes = {}
    for id in range(node_count):
        attributes[id] = {'IPT': rng.randrange(*IPT_range)*IPT_mag,
                          'RAM': rng.randrange(*RAM_range),
                          'storage': rng.randrange(*storage_range)}
    nx.set_node_attributes(graph, attributes)
    
    # Generate and set edge attributes
    edge_attribute = {}
    for edge in graph.edges:
        edge_attribute[edge] = {'BW': rng.randrange(*BW_range),
                                'PR': rng.randrange(*PR_range)}
        edge_attribute[edge]['latency_measure'] = default_packet_size / edge_attribute[edge]['BW'] + edge_attribute[edge]['PR']
        edge_attribute[edge]['speed_measure'] = 1 / edge_attribute[edge]['latency_measure']
    nx.set_edge_attributes(graph, edge_attribute)

    return graph


def convert_dag_to_dic(dag, id):
    '''
    A simple converter function for the sake of compatibility. This function converts networkx DAGs to a YAFS compatible
    python dictionary you can then save those dictionaries in a json file and use them as application definition in YAFS.
    '''
    # References for easier dictionary update
    modules = []
    messages = []
    transmissions = []
    # Base dictionary declaration
    result = {'name':dag.name, 'id':id, 'transmission': transmissions, 'message':messages, 'module':modules}

    # Handling modules and transmissions
    for id, attributes in dag.nodes(data=True):
        # Modules
        modules.append(attributes)
        # Transmissions
        if attributes['Type'] == 'MODULE': # Normal modules have both message_in and message_out parameter
            for *_, message_in in dag.in_edges(id, data='name'):
                for *_, message_out in dag.out_edges(id, data='name'):   
                    transmissions.append({'module': attributes['name'], 'message_in': message_in, 'message_out': message_out})
        elif attributes['Type'] == 'SINK':  # Sinks only have message_in
            for *_, message_in in dag.in_edges(id, data='name'):
                transmissions.append({'module': attributes['name'], 'message_in':message_in})

    # Handling messages
    for *_, edge in dag.edges(data=True):
        messages.append(edge)
    
    return result


def convert_network_to_dic(network_graph):
    '''
    A simple converter function for the sake of compatibility. This function converts networkx graphs to a YAFS compatible
    python dictionary you can then save those dictionaries in a json file and use them as network definition in YAFS.
    '''
    # References for easier dictionary update
    links = []
    entities = []
    # Base dictionary declaration
    result = {'link':links, 'entity': entities}

    # Handling Entities
    for id, attributes in network_graph.nodes(data=True):
        entities.append({'id': id, **attributes})

    # Handling Links
    for s, d, attributes in network_graph.edges(data=True):
        links.append({'s': s, 'd': d, **attributes})
    
    return result


def draw_communities(graph:nx.Graph, communities:list[list[int]]):
    '''
    A tiny little visualiser you can use to visualize communities. For 'communities' parameter use the output from built-in
    partitioning functions of networkx.
    '''
    colors = ('#F5DFB3', '#FFC18C', '#D989B8', '#8268AB', '#7D93B0', '#9DC2BE', '#6DA394', '#D46853', '#344B61', '#E6C363')[:len(communities)]
    # Compute positions for the node clusters as if they were themselves nodes in a
    # super-graph using a larger scale factor
    supergraph = nx.cycle_graph(len(communities))
    superpos = nx.circular_layout(supergraph, scale=len(communities)*2+2)

    # Use the "super-node" positions as the center of each node cluster
    centers = list(superpos.values())
    pos = {}
    for center, comm in zip(centers, communities):
        pos.update(nx.spring_layout(nx.subgraph(graph, comm), center=center, scale=3))

    # Nodes coloured by cluster
    for nodes, clr in zip(communities, colors):
        nx.draw_networkx_nodes(graph, pos=pos, nodelist=nodes, node_color=clr)
    nx.draw_networkx_labels(graph, pos=pos)
    nx.draw_networkx_edges(graph, alpha=0.5, edge_color='tab:gray', pos=pos, connectionstyle='arc3,rad=0.5')

    plt.show()


def draw_communities_with_shape(graph:nx.Graph, communities:list[list[int]], path=None):
    '''
    A tiny little visualiser you can use to visualize communities. For 'communities' parameter use the output from built-in
    partitioning functions of networkx.
    '''
    shapes = 'sodph8'[:len(communities)]
    # Compute positions for the node clusters as if they were themselves nodes in a
    # super-graph using a larger scale factor
    supergraph = nx.cycle_graph(len(communities))
    superpos = nx.circular_layout(supergraph, scale=len(communities)*2+2)

    # Use the "super-node" positions as the center of each node cluster
    centers = list(superpos.values())
    pos = {}
    for center, comm in zip(centers, communities):
        pos.update(nx.circular_layout(nx.subgraph(graph, comm), center=center, scale=3))

    # Nodes coloured by cluster
    for nodes, shp in zip(communities, shapes):
        nx.draw_networkx_nodes(graph, pos=pos, nodelist=nodes, node_color='w', edgecolors='k', node_shape=shp, node_size=350)
    nx.draw_networkx_labels(graph, pos=pos)
    nx.draw_networkx_edges(graph, alpha=0.5, edge_color='tab:gray', pos=pos, connectionstyle='arc3,rad=0.5')

    if path:
        plt.savefig(path, transparent=True)
    else:
        plt.show()
