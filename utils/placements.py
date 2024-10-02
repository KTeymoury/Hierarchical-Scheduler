from yafs.placement import Placement


class MappedPlacement(Placement):
    """
    A static placement class placing modules over devices based on a global dictionary, initial_map.
    """
    def __init__(self, initial_map:dict[str, list[tuple[str, int]]], **kwargs):
        '''
        'initial_map' is a dictionary of structure, {app_name: [(module_name, device_id), ...]}.

        Example
        -------
        initial_map = {application_0: [(Module_0, 12), (Module_1, 12), (Module_2, 6), (Module_2, 4), (Module_3, 1)]}
        placement = MappedPlacement(initial_map)
        '''
        super(MappedPlacement, self).__init__(**kwargs)
        self.initial_map = initial_map

    def initial_allocation(self, sim, app_name):
        services = sim.apps[app_name].services

        if not app_name in self.initial_map:
            raise Exception(f'APPLICATION {app_name} WAS NOT PLACED!!!')
        
        for module, node in self.initial_map[app_name]:
            idDES = sim.deploy_module(app_name, module, services[module], [node])





