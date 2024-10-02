import random
import warnings
from collections import OrderedDict
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scenario import Scenario


class YAFSEnv(gym.Env):
    """Custom YAFS environment that generates a app to partition assignment map and evaluates it
    using YAFS simulator.
    

    Action space is an array of length <PARTITION_COUNT> denoting assignment of current app to
    a partition with a 1.

    Reward is dynamically calculated using YAFS simulations. default violation penalty is -9999.5
    for not assigning an app to any partition and <VIOLATION_PENALTY> for over packing a partition.

    Number of tasks, and, partitions along with their information are passed to environment through
    config dictionary. a parent_scenario field is also there to keep a reference of simulator interface.
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        self.task_count = config.get("task_count", 3)
        self.partition_count = config.get("partition_count", 5)
        self.violation_penalty = config.get("violation_penalty", -0.5)
        self.parent_scenario: Scenario = config.get("parent_scenario", None)
        if self.parent_scenario.prepare_simulation():
            warnings.warn('Scenario was in a simulation ready state.')

        # Future stuff
        # IMPORTANT!!!
        # CPU IS ALWAYS AT INDEX 0 AND RAM IS 1
        tmp_partition_info = np.array([[1000000, 20000],[4000000, 200000],[1000000, 300000],[4000000, 200000],[1000000, 300000]], dtype=np.int64)
        self.partition_info = config.get("partition_info", tmp_partition_info)
        tmp_task_info = np.array([[4000000, 2000, 0],[5000000, 20000, 1],[1000000, 3000, 2],[0, 0, 3]], dtype=np.int64)
        self.task_info = config.get("task_info", tmp_task_info)
        self.partition_usage = np.zeros(shape=(self.partition_count, 2), dtype=np.int64)
        self.cur_task = 0
        self.cur_map = np.zeros(shape=(self.task_count, self.partition_count), dtype=np.int8)

        # NOT SUPPORTED BY RLLIB!!!
        # self.action_space = spaces.MultiBinary(self.partition_count)
        self.action_space = spaces.Box(0, 1, shape=(self.partition_count,), dtype=np.int8)
        # spaces.MultiBinary([self.task_count, self.partition_count]
        self.observation_space = spaces.Dict({"Map": spaces.Box(0, 1, shape=(self.task_count, self.partition_count), dtype=np.int8),
                                              "Task": spaces.Box(low=0, high=np.iinfo(np.int64).max-1, shape=(3,), dtype=np.int64)}, seed=5)
        
    def reset(self, *, seed=None, options=None):
        random.seed(seed)
        self.cur_task = 0
        self.cur_map[:,:] = 0
        self.partition_usage[:,:] = 0
        # Return obs and (empty) info dict.
        return (OrderedDict([('Map', self.cur_map), ('Task', self.task_info[0])]), {})

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        infos = {'violation': None}

        # Check the assignment to make sure a task is assigned to at least one partition
        if not action.any():
            infos['violation'] = 'Did not assign task'
            reward = -9999.5
            terminated = True

        # Loosely check if the list of assigned tasks fit in the partition
        elif self.__constraint_check(self.task_info[self.cur_task][:1], np.nonzero(action)[0]):
            self.cur_map[self.cur_task] = action
            self.cur_task += 1
            terminated = self.cur_task >= self.task_count

            # Run a simulation and retrieve reward from YAFS at the end of each episode
            reward = self.parent_scenario.test_partition_assignment(self.cur_map, True) if terminated else 0
        
        else:
            infos['violation'] = 'Violated hard constraint'
            reward = self.violation_penalty
            terminated = True
        
        return (
            OrderedDict([('Map', self.cur_map), ('Task', self.task_info[self.task_count])]),
            reward,
            terminated,
            truncated,
            infos,
        )
    
    def __constraint_check(self, task, partitions):
        for id in partitions:
            self.partition_usage[id] += task
            # If you wanted to check CPU too use this:
            # self.partition_usage[id][0] > partition_info[id][0]
            if self.partition_usage[id][1] > self.partition_info[id][1]:
                return False
        return True
