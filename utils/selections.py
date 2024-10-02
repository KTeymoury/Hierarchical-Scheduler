import networkx as nx
from yafs.selection import Selection


# If you set this flag common stuff will be cached session wise
USE_COMMONS = True
_common_set = False
# Common stuff :)
COMMON_PATH_CACHE = None
COMMON_PARTITION_TO_INFO = None
COMMON_DEVICE_TO_PARTITION = None
COMMON_GATEWAYS = None


def _initialize_commons(path_cache, partition_info, gateways):
    '''
    Internal function to speed that initiates a module wide cache to speed things up.
    '''
    global _common_set, COMMON_PATH_CACHE, COMMON_PARTITION_TO_INFO, COMMON_DEVICE_TO_PARTITION, COMMON_GATEWAYS
    COMMON_PATH_CACHE = path_cache
    dev_to_part = {}
    COMMON_PARTITION_TO_INFO = []
    for i in range(len(partition_info)):
        partition = partition_info[i]
        COMMON_PARTITION_TO_INFO.append((partition['SUBGRAPH'], partition['CENTER']))
        for device in partition['DEVICES']:
            dev_to_part[device] = i
    COMMON_PARTITION_TO_INFO = tuple(COMMON_PARTITION_TO_INFO)
    COMMON_DEVICE_TO_PARTITION = tuple([i for k,i in sorted(dev_to_part.items())])
    COMMON_GATEWAYS = set(gateways)
    _common_set = True



# TODO: Make this happen T_T
class CongestionAwarePartitionedSelector(Selection):
    pass


class ClosestFirstPartitionedSelector(Selection):
    '''
    A simple selector that routes requests to closest partition with an instance of that app
    '''
    def __init__(self, gateways, partition_info, path_cache = {}, logger=None):
        '''
        This selector needs to know the name of gateway devices and general information about partitions in order
        to operate properly. You can find a usage example inside 'Scenario.py'.
        '''
        super().__init__(logger)

        # Mostly static stuff
        self.partition_info: tuple[nx.Graph, int] = None
        self.device_to_partition: tuple[int] = None
        self.cache: dict[tuple[int|str, int|str], tuple[tuple[int|str]]] = None
        self.gateways: set[str] = None

        # Dynamic stuff
        self.DES_cache: dict[tuple[str, str], dict[int, tuple[int|str, int]]] = {}
        self.round_robin: dict[str, int] = {}
        self.app_to_partitions: dict[str, tuple[int]] = {}

        # Commons go brrrrrr
        if USE_COMMONS:
            if not _common_set:
                raise Exception('Initialize commons before using them.')
            self.partition_info = COMMON_PARTITION_TO_INFO
            self.device_to_partition = COMMON_DEVICE_TO_PARTITION
            self.cache = COMMON_PATH_CACHE
            self.gateways = COMMON_GATEWAYS
        # Generate stuff for this instance instead of using commons
        else:
            dev_to_part = {}
            self.partition_info = []
            for i in range(len(partition_info)):
                partition = partition_info[i]
                self.partition_info.append((partition['SUBGRAPH'], partition['CENTER']))
                for device in partition['DEVICES']:
                    dev_to_part[device] = i
            self.partition_info = tuple(self.partition_info)

            self.cache = path_cache
            self.gateways = set(gateways)
            self.device_to_partition = tuple([i for k,i in sorted(dev_to_part.items())])


    def get_path(self, sim, app_name, message, topology_src, alloc_DES, alloc_module, traffic, from_des):
        """
        Returns the Simpy ID of first instance of target module along with the topological path to the network node
        on which that module is placed.
        """
        # Index the app if it's first time seeing it (This also caches first module of this app)
        if not (app_name in self.app_to_partitions):
            targets = []
            for DES_destination in alloc_module[app_name][message.dst]:
                node_destination = alloc_DES[DES_destination]
                partition_number = self.device_to_partition[node_destination]
                targets.append(partition_number)
                self.DES_cache[app_name, message.dst, partition_number] = (DES_destination, node_destination)
            targets = tuple(targets)
            self.app_to_partitions[app_name] = (targets, len(targets))

        # Now we select a target partition based on round robin method if it's a source message
        target_partition = 0
        if message.src.startswith('Source'):
            targets, length = self.app_to_partitions[app_name]
            current = self.round_robin.get(app_name, 0)
            target_partition = targets[current]
            self.round_robin[app_name] = (current + 1) % length
        # If it's not a source message pick whichever partition it originates from
        else:
            target_partition = self.device_to_partition[topology_src]
        
        # We use a DES cache to store simpy ids and topology nodes for every module on all of the partitions.
        # Update the cache if it's our first time seeing this target module.
        ## NOTE: should I really cache this thing?
        if not ((app_name, message.dst, target_partition) in self.DES_cache):
            for DES_destination in alloc_module[app_name][message.dst]:
                node_destination = alloc_DES[DES_destination]
                partition_number = self.device_to_partition[node_destination]
                self.DES_cache[app_name, message.dst, partition_number] = (DES_destination, node_destination)
        
        # Get simpy id and topology destination based for the module on target partition
        DES_destination, node_destination = self.DES_cache[app_name, message.dst, target_partition]

        # Alias this thing for backwards compatibility
        node_source = topology_src

        # These are our final results:
        bestDES = (DES_destination,)
        bestPath = None

        # We'll handle the path generation in this section
        ## Short circuit to save memory on cache (also a bit faster)
        if node_destination == node_source:
            bestPath = ((node_destination,),)
        ## Retrieve path from cache if there is any
        elif (node_source, node_destination) in self.cache:
            bestPath = self.cache[node_source, node_destination]
        ## Calculate a path and return after caching if all fails
        else:
            path = None
            subgraph, center = self.partition_info[target_partition]
            # Gateways are a bit unique because of their string ids so we handle them separately
            if node_source in self.gateways:
                path = [node_source] + list(nx.shortest_path(subgraph, source=center, target=node_destination, weight='latency_measure'))
                path = tuple(path)
            else:
                path = tuple(nx.shortest_path(subgraph, source=node_source, target=node_destination, weight='latency_measure'))

            bestPath = (path,)
            self.cache[node_source, node_destination] = bestPath

        return bestPath, bestDES


class RoundRobinPartitionedSelector(Selection):
    '''
    A simple round robin selection class that routes requests to partitions using round robin technique.
    As a selector this is nt very useful but it's going to be the foundation of congestion-aware one soon:TM:.
    '''
    def __init__(self, gateways, partition_info, path_cache = {}, logger=None):
        '''
        This selector needs to know the name of gateway devices and general information about partitions in order
        to operate properly. You can find a usage example inside 'Scenario.py'.
        '''
        super().__init__(logger)

        # Mostly static stuff
        self.partition_info: tuple[nx.Graph, int] = None
        self.device_to_partition: tuple[int] = None
        self.cache: dict[tuple[int|str, int|str], tuple[tuple[int|str]]] = None
        self.gateways: set[str] = None

        # Dynamic stuff
        self.DES_cache: dict[tuple[str, str], dict[int, tuple[int|str, int]]] = {}
        self.round_robin: dict[str, int] = {}
        self.app_to_partitions: dict[str, tuple[int]] = {}

        # Commons go brrrrrr
        if USE_COMMONS:
            if not _common_set:
                raise Exception('Initialize commons before using them.')
            self.partition_info = COMMON_PARTITION_TO_INFO
            self.device_to_partition = COMMON_DEVICE_TO_PARTITION
            self.cache = COMMON_PATH_CACHE
            self.gateways = COMMON_GATEWAYS
        # Generate stuff for this instance instead of using commons
        else:
            dev_to_part = {}
            self.partition_info = []
            for i in range(len(partition_info)):
                partition = partition_info[i]
                self.partition_info.append((partition['SUBGRAPH'], partition['CENTER']))
                for device in partition['DEVICES']:
                    dev_to_part[device] = i
            self.partition_info = tuple(self.partition_info)

            self.cache = path_cache
            self.gateways = set(gateways)
            self.device_to_partition = tuple([i for k,i in sorted(dev_to_part.items())])


    def get_path(self, sim, app_name, message, topology_src, alloc_DES, alloc_module, traffic, from_des):
        """
        Returns the Simpy ID of first instance of target module along with the topological path to the network node
        on which that module is placed.
        """
        # Index the app if it's first time seeing it (This also caches first module of this app)
        if not (app_name in self.app_to_partitions):
            targets = []
            for DES_destination in alloc_module[app_name][message.dst]:
                node_destination = alloc_DES[DES_destination]
                partition_number = self.device_to_partition[node_destination]
                targets.append(partition_number)
                self.DES_cache[app_name, message.dst, partition_number] = (DES_destination, node_destination)
            targets = tuple(targets)
            self.app_to_partitions[app_name] = (targets, len(targets))

        # Now we select a target partition based on round robin method if it's a source message
        target_partition = 0
        if message.src.startswith('Source'):
            targets, length = self.app_to_partitions[app_name]
            current = self.round_robin.get(app_name, 0)
            target_partition = targets[current]
            self.round_robin[app_name] = (current + 1) % length
        # If it's not a source message pick whichever partition it originates from
        else:
            target_partition = self.device_to_partition[topology_src]
        
        # We use a DES cache to store simpy ids and topology nodes for every module on all of the partitions.
        # Update the cache if it's our first time seeing this target module.
        ## NOTE: should I really cache this thing?
        if not ((app_name, message.dst, target_partition) in self.DES_cache):
            for DES_destination in alloc_module[app_name][message.dst]:
                node_destination = alloc_DES[DES_destination]
                partition_number = self.device_to_partition[node_destination]
                self.DES_cache[app_name, message.dst, partition_number] = (DES_destination, node_destination)
        
        # Get simpy id and topology destination based for the module on target partition
        DES_destination, node_destination = self.DES_cache[app_name, message.dst, target_partition]

        # Alias this thing for backwards compatibility
        node_source = topology_src

        # These are our final results:
        bestDES = (DES_destination,)
        bestPath = None

        # We'll handle the path generation in this section
        ## Short circuit to save memory on cache (also a bit faster)
        if node_destination == node_source:
            bestPath = ((node_destination,),)
        ## Retrieve path from cache if there is any
        elif (node_source, node_destination) in self.cache:
            bestPath = self.cache[node_source, node_destination]
        ## Calculate a path and return after caching if all fails
        else:
            path = None
            subgraph, center = self.partition_info[target_partition]
            # Gateways are a bit unique because of their string ids so we handle them seperately
            if node_source in self.gateways:
                path = [node_source] + list(nx.shortest_path(subgraph, source=center, target=node_destination, weight='latency_measure'))
                path = tuple(path)
            else:
                path = tuple(nx.shortest_path(subgraph, source=node_source, target=node_destination, weight='latency_measure'))

            bestPath = (path,)
            self.cache[node_source, node_destination] = bestPath

        return bestPath, bestDES
    


class PartitionedSelector(Selection):
    '''
    Nothing fancy... just a selector optimized to be used in partitioned networks. Basically just a performant version of
    cached selector.
    '''
    def __init__(self, gateways, partition_info, path_cache = {}, logger=None):
        '''
        This selector needs to know the name of gateway devices and general information about partitions in order
        to operate properly. You can find a usage example inside 'Scenario.py'.
        '''
        super().__init__(logger)

        # Statics
        self.partition_info: tuple[nx.Graph, int] = None
        self.device_to_partition: tuple[int] = None
        self.cache: dict[tuple[int|str, int|str], tuple[tuple[int|str]]] = None
        self.gateways: set[str] = None

        # Commons go brrrrrr
        if USE_COMMONS:
            if not _common_set:
                raise Exception('Initialize commons before using them.')
            self.partition_info = COMMON_PARTITION_TO_INFO
            self.device_to_partition = COMMON_DEVICE_TO_PARTITION
            self.cache = COMMON_PATH_CACHE
            self.gateways = COMMON_GATEWAYS
        # Generate stuff for this instance instead of using commons
        else:
            dev_to_part = {}
            self.partition_info = []
            for i in range(len(partition_info)):
                partition = partition_info[i]
                self.partition_info.append((partition['SUBGRAPH'], partition['CENTER']))
                for device in partition['DEVICES']:
                    dev_to_part[device] = i
            self.partition_info = tuple(self.partition_info)

            self.cache = path_cache
            self.gateways = set(gateways)
            self.device_to_partition = tuple([i for k,i in sorted(dev_to_part.items())])


    def get_path(self, sim, app_name, message, topology_src, alloc_DES, alloc_module, traffic,from_des):
        """
        Returns the Simpy ID of first instance of target module along with the topological path to the network node
        on which that module is placed.
        """
        node_source = topology_src

        # Gives a list of simpy IDs for each module
        DES_destinations = alloc_module[app_name][message.dst]
        bestPath = None
        bestDES = None
        for DES_destination in DES_destinations: ## In this case, there are only one deployment
            node_destination = alloc_DES[DES_destination]

            if node_destination == node_source: # Short circuit to save memory on cache
                bestPath = ((node_destination,),)
                bestDES = (DES_destination,)
            elif (node_source, node_destination) in self.cache:
                bestPath = self.cache[node_source, node_destination]
                bestDES = (DES_destination,)
            else:
                path = None
                target_partition = self.device_to_partition[node_destination]
                subgraph, center = self.partition_info[target_partition]
                if node_source in self.gateways:
                    path = [node_source] + list(nx.shortest_path(subgraph, source=center, target=node_destination, weight='latency_measure'))
                    path = tuple(path)
                elif self.device_to_partition[node_source] == self.device_to_partition[node_destination]:
                    path = tuple(nx.shortest_path(subgraph, source=node_source, target=node_destination, weight='latency_measure'))
                else:
                    continue
                bestPath = (path,)
                self.cache[node_source, node_destination] = bestPath
                bestDES = (DES_destination,)

            return bestPath, bestDES


class CachedSelector(Selection):
    '''
    A simple selector class that uses a path lookup cache to speed up simulation.
    '''
    def __init__(self, path_cache = {}, logger=None):
        super().__init__(logger)
        self.cache: dict[tuple[int|str, int|str], tuple[tuple[int|str]]] = path_cache
        
        # Commons go brrrrrr
        if USE_COMMONS:
            if not _common_set:
                raise Exception('Initialize commons before using them.')
            self.cache = COMMON_PATH_CACHE


    def get_path(self, sim, app_name, message, topology_src, alloc_DES, alloc_module, traffic,from_des):
        """
        Returns the Simpy ID of first instance of target module along with the topological path to the network node
        on which that module is placed.
        """
        node_source = topology_src

        # Gives a list of simpy IDs for each module
        DES_destinations = alloc_module[app_name][message.dst]
        bestPath = []
        bestDES = []
        for DES_destination in DES_destinations: ## In this case, there are only one deployment
            node_destination = alloc_DES[DES_destination]

            if node_destination == node_source: # Short circuit to save memory on cache
                bestPath = ((node_destination,),)
                bestDES = (DES_destination,)
            elif (node_source, node_destination) in self.cache:
                bestPath = self.cache[node_source, node_destination]
                bestDES = (DES_destination,)
            else:
                bestPath = (tuple(nx.shortest_path(sim.topology.G, source=node_source, target=node_destination, weight='latency_measure')),)
                self.cache[node_source, node_destination] = bestPath
                bestDES = (DES_destination,)

            return bestPath, bestDES

