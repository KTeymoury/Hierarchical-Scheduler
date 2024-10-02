import networkx as nx
from operator import itemgetter
from ortools.linear_solver import pywraplp
import numpy as np


def ilp_placement(partition: nx.Graph, apps: list[nx.DiGraph]) -> tuple[bool, bool, dict]:
    '''
    An ILP solution that places given applications on physical devices inside provided partition.
    This function uses ortools and can take a while to finish executing, so please be patient.
    Final output is a 3-tuple where first two elements tell us if everything was placed and whether an optimal
    solution was reached or not and the third element is the placement map compatible with 'MappedPlacement'.
    '''
    # Final results
    placed_all = True
    optimal = True
    result = {}

    # Initialize partition information
    device_free_ram = []
    device_IPT = []
    device_ids = []
    device_count = 0

    # Get shortest path costs for each node pair
    network_cost = dict(nx.shortest_path_length(partition, weight='latency_measure'))

    # Get Ram and IPT info for each node
    for device_id, attribs in sorted(partition.nodes(data=True), key=itemgetter(0)):
        device_ids.append(device_id)
        device_free_ram.append(attribs['RAM'])
        device_IPT.append(attribs['IPT'])
    
    # Make an id mapping from actual device ids to local ones
    reverse_device_ids = {device_ids[i]:i for i in range(len(device_ids))}

    # Re-map network cost to local node names
    remapped_network_cost = {}
    for source in network_cost:
        for target in network_cost[source]:
            remapped_network_cost[reverse_device_ids[source], reverse_device_ids[target]] = network_cost[source][target]

    # Update device count
    device_count = len(device_ids)

    # Convert information to numpy arrays for efficient access
    device_IPT = np.array(device_IPT, dtype=np.float64)
    device_IPT.setflags(write=False) # Lock the IPT array
    device_free_ram = np.array(device_free_ram, dtype=np.int64)
    device_ids = tuple(device_ids)

    for app in apps:
        # Get service informations for each service in an app
        service_ram = tuple([ram for _, ram in sorted(app.nodes(data='RAM'), key=itemgetter(0))[1:]])
        # Update service count
        service_count = len(service_ram)
        
        # Initialize processing cost with each messages instruction count
        process_costs = np.zeros(shape=(service_count, device_count), dtype=np.float64)
        for _, service_id, instructions in app.edges(data='instructions'):
            process_costs[service_id-1] = instructions
        
        # Update processing cost on each device
        process_costs /= device_IPT
        process_costs.setflags(write=False) # lock the cost array

        # Solver
        # Create the mip solver with the SCIP backend.
        solver = pywraplp.Solver.CreateSolver("SCIP")
        solver.Clear()
        if not solver:
            raise Exception('Could NOT make a solver instance!')

        # Variables
        # x[service_id, device_id] is an array of 0-1 variables, which will be 1
        # if service 'service_id' is assigned to device 'device_id'.
        x = {}
        for service_id in range(service_count):
            for device_id in range(device_count):
                x[service_id, device_id] = solver.BoolVar(f"x[{service_id},{device_id}]")
        
        # Constraints
        # Sum of ram of services assigned to a device should be lesser that its ram.
        for device_id in range(device_count):
            solver.Add(sum([service_ram[service_id] * x[service_id, device_id] for service_id in range(service_count)]) <= device_free_ram[device_id], f'TotalRamFor_{device_id}')

        # Each service is assigned to exactly one device.
        for service_id in range(service_count):
            solver.Add(solver.Sum([x[service_id, device_id] for device_id in range(device_count)]) == 1, f'SingleService_{service_id}')
        
        # Objectives
        # Summing up process times
        objective_terms = []
        for service_id in range(service_count):
            for device_id in range(device_count):
                objective_terms.append(process_costs[service_id, device_id] * x[service_id, device_id])
        
        # Summing up the link latencies IF any
        links = {}
        link_count = 0
        for source_service, target_service, size in app.edges(data='bytes'):
            # TODO: use 'size' to calculate actual latency
            if source_service == 0:
                continue
            source_service -= 1
            target_service -= 1
            for source_device in range(device_count):
                for target_device in range(device_count):
                    if source_device == target_device:
                        continue
                    link_count+=1
                    link = solver.BoolVar(f"transmit_{source_service}_{target_service}__{source_device}_{target_device}")
                    links[source_service, target_service, source_device, target_device] = link
                    solver.Add(link <= x[source_service, source_device], f'link_def_{link_count}_1')
                    solver.Add(link <= x[target_service, target_device], f'link_def_{link_count}_2')
                    solver.Add(link >= x[source_service, source_device] + x[target_service, target_device] - 1, f'link_def_{link_count}_3')
                    objective_terms.append(remapped_network_cost[source_device, target_device] * link)

        # Setting up the objective function
        solver.Minimize(solver.Sum(objective_terms))

        # IMPORTANT: Use the second one if you want to export or import
        '''
        with open('./lp_file.lp', 'w') as f:
            f.write(solver.ExportModelAsLpFormat(obfuscated=False))
        with open('./mps_file.mps', 'w') as f:
            f.write(solver.ExportModelAsMpsFormat(obfuscated=False, fixed_format=False))
        '''

        # Solve
        # print(f"Solving for {app.name} with {solver.SolverVersion()}\n")
        status = solver.Solve()
        
        # Print solution.
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            optimal = optimal and pywraplp.Solver.OPTIMAL
            service_name = dict(app.nodes(data='name'))
            
            placemnt_map = []
            for service_id, device_id in x:
                # If service is placed on a device...
                if x[service_id, device_id].solution_value() > 0.5:
                    # Update remaining free ram for that device
                    device_free_ram[device_id] -= service_ram[service_id]
                    # Add placement to dictionary
                    placemnt_map.append((service_name[service_id+1], device_ids[device_id]))
            result[app.name] = tuple(placemnt_map)
        else:
            placed_all = False
            optimal = False
    
    return placed_all, optimal, result


def ilp_placement_source_included(partition: nx.Graph, entry_point_id: int, apps: list[nx.DiGraph]) -> tuple[bool, bool, dict]:
    '''
    An ILP solution that places given applications on physical devices inside provided partition.
    This function uses ortools and can take a while to finish executing, so please be patient.
    Final output is a 3-tuple where first two elements tell us if everything was placed and whether an optimal
    solution was reached or not and the third element is the placement map compatible with 'MappedPlacement'.
    '''
    # Final results
    placed_all = True
    optimal = True
    result = {}

    # Initialize partition information
    device_free_ram = []
    device_IPT = []
    device_ids = []
    device_count = 0

    # Get shortest path costs for each node pair
    network_cost = dict(nx.shortest_path_length(partition, weight='latency_measure'))

    # Get Ram and IPT info for each node
    for device_id, attribs in sorted(partition.nodes(data=True), key=itemgetter(0)):
        device_ids.append(device_id)
        device_free_ram.append(attribs['RAM'])
        device_IPT.append(attribs['IPT'])
    
    # Make an id mapping from actual device ids to local ones
    reverse_device_ids = {device_ids[i]:i for i in range(len(device_ids))}
    new_entry_point_id  = reverse_device_ids[entry_point_id]
    # Re-map network cost to local node names
    remapped_network_cost = {}
    for source in network_cost:
        for target in network_cost[source]:
            remapped_network_cost[reverse_device_ids[source], reverse_device_ids[target]] = network_cost[source][target]

    # Update device count
    device_count = len(device_ids)

    # Convert information to numpy arrays for efficient access
    device_IPT = np.array(device_IPT, dtype=np.float64)
    device_IPT.setflags(write=False) # Lock the IPT array
    device_free_ram = np.array(device_free_ram, dtype=np.int64)
    device_ids = tuple(device_ids)

    for app in apps:
        # Get service informations for each service in an app
        service_ram = tuple([ram for _, ram in sorted(app.nodes(data='RAM'), key=itemgetter(0))[1:]])
        # Update service count
        service_count = len(service_ram)
        
        # Initialize processing cost with each messages instruction count
        process_costs = np.zeros(shape=(service_count, device_count), dtype=np.float64)
        for _, service_id, instructions in app.edges(data='instructions'):
            process_costs[service_id-1] = instructions
        
        # Update processing cost on each device
        process_costs /= device_IPT
        process_costs.setflags(write=False) # lock the cost array

        # Solver
        # Create the mip solver with the SCIP backend.
        solver = pywraplp.Solver.CreateSolver("SCIP")
        solver.Clear()
        if not solver:
            raise Exception('Could NOT make a solver instance!')

        # Variables
        # x[service_id, device_id] is an array of 0-1 variables, which will be 1
        # if service 'service_id' is assigned to device 'device_id'.
        x = {}
        for service_id in range(service_count):
            for device_id in range(device_count):
                x[service_id, device_id] = solver.BoolVar(f"x[{service_id},{device_id}]")
        
        # Constraints
        # Sum of ram of services assigned to a device should be lesser that its ram.
        for device_id in range(device_count):
            solver.Add(sum([service_ram[service_id] * x[service_id, device_id] for service_id in range(service_count)]) <= device_free_ram[device_id], f'TotalRamFor_{device_id}')

        # Each service is assigned to exactly one device.
        for service_id in range(service_count):
            solver.Add(solver.Sum([x[service_id, device_id] for device_id in range(device_count)]) == 1, f'SingleService_{service_id}')
        
        # Objectives
        # Summing up process times
        objective_terms = []
        for service_id in range(service_count):
            for device_id in range(device_count):
                objective_terms.append(process_costs[service_id, device_id] * x[service_id, device_id])
        
        # Summing up the link latencies IF any
        links = {}
        link_count = 0
        for source_service, target_service, size in app.edges(data='bytes'):
            # TODO: use 'size' to calculate actual latency
            if source_service == 0:
                target_service -= 1
                for target_device in range(device_count):
                    if target_device == new_entry_point_id:
                        continue
                    objective_terms.append(remapped_network_cost[new_entry_point_id, target_device] * x[target_service, target_device])
                continue
            source_service -= 1
            target_service -= 1
            for source_device in range(device_count):
                for target_device in range(device_count):
                    if source_device == target_device:
                        continue
                    link_count+=1
                    link = solver.BoolVar(f"transmit_{source_service}_{target_service}__{source_device}_{target_device}")
                    links[source_service, target_service, source_device, target_device] = link
                    solver.Add(link <= x[source_service, source_device], f'link_def_{link_count}_1')
                    solver.Add(link <= x[target_service, target_device], f'link_def_{link_count}_2')
                    solver.Add(link >= x[source_service, source_device] + x[target_service, target_device] - 1, f'link_def_{link_count}_3')
                    objective_terms.append(remapped_network_cost[source_device, target_device] * link)

        # Setting up the objective function
        solver.Minimize(solver.Sum(objective_terms))

        # IMPORTANT: Use the second one if you want to export or import
        '''
        with open('./lp_file.lp', 'w') as f:
            f.write(solver.ExportModelAsLpFormat(obfuscated=False))
        with open('./mps_file.mps', 'w') as f:
            f.write(solver.ExportModelAsMpsFormat(obfuscated=False, fixed_format=False))
        '''

        # Solve
        # print(f"Solving for {app.name} with {solver.SolverVersion()}\n")
        status = solver.Solve()
        
        # Print solution.
        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            optimal = optimal and pywraplp.Solver.OPTIMAL
            service_name = dict(app.nodes(data='name'))
            
            placemnt_map = []
            for service_id, device_id in x:
                # If service is placed on a device...
                if x[service_id, device_id].solution_value() > 0.5:
                    # Update remaining free ram for that device
                    device_free_ram[device_id] -= service_ram[service_id]
                    # Add placement to dictionary
                    placemnt_map.append((service_name[service_id+1], device_ids[device_id]))
            result[app.name] = tuple(placemnt_map)
        else:
            placed_all = False
            optimal = False
    
    return placed_all, optimal, result


def greedy_placement(partition: nx.Graph, apps: list[tuple[str, int, int, tuple[str, int, int]]]) -> tuple[bool, bool, dict]:
    '''
    A simple and fast greedy placement technique used in training phase for faster execution. it receives applications in a
    different format. please refer to 'Scenario.py' for usage example.
    Final output is a 3-tuple where first two elements tell us if everything was placed and whether a 'nice' solution was
    reached or not and the third element is the placement map compatible with 'MappedPlacement'.
    '''
    # End results
    result = {app[0]:[] for app in apps}
    placed_all = True
    nice_placement = True
    
    # Partition information
    tmp_nodes = partition.nodes(data=True)
    devices = []
    min_ipt = min([attributes['IPT'] for _, attributes in tmp_nodes])
    for device_id, attributes in partition.nodes(data=True):
        # List[Device_ID, IPT, Available_RAM, Priority]
        devices.append([device_id, attributes['IPT'], attributes['RAM'], int(attributes['IPT']/min_ipt + 0.2)])
    devices.sort(key=itemgetter(1, 2), reverse=True)
    candidate_devices = devices.copy()

    # Sort applications based on instructions
    remaining_apps = list(apps) if type(apps) is tuple else apps.copy()
    remaining_apps.sort(key=itemgetter(1,2), reverse=True)
    remaining_app_count = len(remaining_apps)
    
    while len(candidate_devices) > 0 and remaining_app_count > 0:
        # Offset to compensate candidate elimination
        offset = 0
        # Loop over candidates based on index
        for device_index in range(len(candidate_devices)):
            # Short circuit condition
            if remaining_app_count == 0:
                break
            # Take into account eliminated candidates to avoid out of bounds
            device_index -= offset
            # Cache the element to avoid list access delay
            device = candidate_devices[device_index]
            # Based on device IPT ratio place at most X apps on it where X is stored in device[3]
            for _ in range(device[3]):
                # Short circuit condition
                if remaining_app_count == 0:
                    break
                # Loop over remaining apps based on index
                for app_index in range(len(remaining_apps)):
                    # Check and see if it fits in device's memory
                    if remaining_apps[app_index][2] < device[2]:
                        # Cache the name to reduce list access
                        app_name = remaining_apps[app_index][0]
                        # Assign every service in this app to current device
                        for service in remaining_apps[app_index][3]:
                            result[app_name].append((service[0], device[0]))
                        # Update remaining RAM on device
                        device[2] -= remaining_apps[app_index][2]
                        # Dequeue app
                        del remaining_apps[app_index]
                        # Update the app counter
                        remaining_app_count -= 1
                        # Get out of app loop and prevent execution of else block
                        break
                # If none of the remaining apps fit on the device
                else:
                    # Eliminate this candidate and offset the index by -1
                    del candidate_devices[device_index]
                    offset += 1
                    break
    
    if remaining_app_count > 0:
        nice_placement = False

        # Unpack services and sort by RAM
        remaining_services = []
        for app in remaining_apps:
            remaining_services.extend([(app[0], *service) for service in app[3]])
        remaining_services.sort(key=itemgetter(2, 3), reverse=True)
        
        # Sort devices by RAM too (this on is in ascending order!)
        devices.sort(key=itemgetter(2))

        # Try and fit largest services in smallest spaces first
        for device in devices:
            # Initialize an offset for item removal
            offset = 0
            for service_index in range(len(remaining_services)):
                # Take index back to avoid out of bound
                service_index -= offset
                # Check and see if this service fits on device
                if remaining_services[service_index][3] < device[2]:
                    # Assign this service to current device
                    result[remaining_services[service_index][0]].append((remaining_services[service_index][1], device[0]))
                    # Update remaining RAM on device
                    device[2] -= remaining_services[service_index][3]
                    # Dequeue service
                    del remaining_services[service_index]
                    # Increase offset to compensate item removal
                    offset += 1
        # Check and see if everything is placed or not
        placed_all = (len(remaining_services) == 0)
    
    return placed_all, nice_placement, result
