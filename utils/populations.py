import networkx as nx
from yafs.population import Population
from yafs.distribution import deterministicDistributionStartPoint


class CompatiblePopulation(Population):
    """
    A modified version of YAFS example population kept here for future reference.
    """
    def initial_allocation(self,sim,app_name):
        for ctrl in self.sink_control:
            sim.deploy_sink(app_name, node=ctrl['id'], module=ctrl['module'])

        for ctrl in self.src_control:
            for i in range(ctrl['number']):
                idsrc = sim.deploy_source(app_name, id_node=ctrl['id'], msg=ctrl['message'], distribution=ctrl['distribution'])
            # the idsrc can be used to control the deactivation of the process in a dynamic behaviour


class MappedPopulation(Population):
    """
    A Static population assigning sources and sinks based on a global dictionary.
    """
    def __init__(self, population_map, **kwargs):
        '''
        population_map should be a dictionary of {app_name: (sources, sinks)}
        sources is a list of [{device_id, count, distribution}] where distribution is a sub-class of YAFS distribution class
        sinks = [{device_id, module_name}]

        Note: You can pass an empty iterable as 'sinks'

        
        Example
        -------
        population_map = {'Application_0': [('GW7', 5, deterministicDistributionStartPoint(0, 5000)),
                                            ('GW2', 2, deterministicDistributionStartPoint(32, 400))
                                            ], []],
                          'Application_1': [('GW3', 25, deterministicDistributionStartPoint(0, 5000))], []]}
        population = MappedPopulation(population_map)
        '''
        self.population_map = population_map
        super().__init__(**kwargs)


    def initial_allocation(self,sim,app_name):
        # A dictionary of actual message objects originating from source modules
        source_messages = sim.apps[app_name].messages

        # Break assignment information into source and sink information
        source_info, sink_info = self.population_map[app_name]

        for message in source_messages:
            # source_info = tuple(device_id, message_count, distribution)
            for _ in range(source_info[1]): 
                idsrc = sim.deploy_source(app_name, id_node=source_info[0], msg=source_messages[message], distribution=source_info[2])
        
        for device_id, module_name in sink_info:
            # sink_info = [tuple(device_id, module_name), ...]
            sim.deploy_sink(app_name, node=device_id, module=module_name)


class MappedPopulationImproved(Population):
    """
    A Static population assigning sources and sinks based on a global dictionary.
    """
    def __init__(self, population_map, **kwargs):
        '''
        population_map should be a dictionary of {app_name: (sources, sinks)}
        sources is a list of [{device_id, count, distribution}] where distribution is a sub-class of YAFS distribution class
        sinks = [{device_id, module_name}]

        Note: You can pass an empty iterable as 'sinks'

        
        Example
        -------
        population_map = {'Application_0': [('GW7', 5, deterministicDistributionStartPoint(0, 5000)),
                                            ('GW2', 2, deterministicDistributionStartPoint(32, 400))
                                            ], []],
                          'Application_1': [('GW3', 25, deterministicDistributionStartPoint(0, 5000))], []]}
        population = MappedPopulation(population_map)
        '''
        self.population_map = population_map
        super().__init__(**kwargs)


    def initial_allocation(self,sim,app_name):
        # A dictionary of actual message objects originating from source modules
        source_messages = sim.apps[app_name].messages

        # Break assignment information into source and sink information
        source_info, sink_info = self.population_map[app_name]

        for message in source_messages:
            # source_info = tuple(device_id, message_count, distribution)
            for _ in range(source_info[1]): 
                idsrc = sim.deploy_source(app_name, id_node=source_info[0],
                                          msg=source_messages[message],
                                          distribution=deterministicDistributionStartPoint(name="Deterministic", time=source_info[2].time, start=source_info[2].start))
        
        for device_id, module_name in sink_info:
            # sink_info = [tuple(device_id, module_name), ...]
            sim.deploy_sink(app_name, node=device_id, module=module_name)