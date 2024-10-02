import statistics
import csv
import warnings
from copy import copy


from yafs.metrics import Metrics

# Default headers in YAFS
YAFS_APP_LOG_HEADERS = ['id', 'type', 'app', 'module', 'message', 'DES.src', 'DES.dst',
                        'TOPO.src', 'TOPO.dst', 'module.src', 'service', 'time_in', 'time_out',
                        'time_emit', 'time_reception']
YAFS_LINK_LOG_HEADERS = ['id', 'type', 'src', 'dst', 'app', 'latency', 'message', 'ctime', 'size', 'buffer']
NO_RESULT_DEFAULT = 9999999.9

class MetricLeaker(Metrics):
    '''
    A simple leaker object that saves every feedback received from YAFS. Final result of this object is a tuple of two lists,
    first one being the application logs and the second one being link logs.
    '''
    def __init__(self):
        self.app_log = []
        self.link_log = []
        self.result = (self.app_log, self.link_log)

    def flush(self):
        pass
    
    def insert(self, value):
        self.app_log.append(copy(value))
        
    def insert_link(self, value):
        self.app_log.append(copy(value))

    def close(self):
        self.result = (self.app_log, self.link_log)


class RunTimeLeaker(Metrics):
    '''
    A useful leaker object for when you need to get the total run time for applications. Final result of this leaker is the
    average run time of every app. Run time takes into account the execution time AND transmission time.
    '''
    def __init__(self):
        self.time = {}
        self.counter = {}
        self.result = None

    def flush(self):
        pass
    
    def insert(self, value):
        app_name = value['app']
        if app_name in self.time:
            self.time[app_name] += (value['time_out'] - value['time_emit'])
            if value['module.src'] == 'Source_0':
                self.counter[app_name] += 1
        else:
            self.time[app_name] = (value['time_out'] - value['time_emit'])
            self.counter[app_name] = 1
        
    def insert_link(self, value):
        pass

    def close(self):
        times = self.time
        count = self.counter
        stats = [times[name]/count[name] for name in times]
        if len(stats) == 0:
            self.result = NO_RESULT_DEFAULT
            warnings.warn('No Results were captured during simulation. Returning default result.')
        else:
            self.result = statistics.mean(stats)


class RunTimeLeakerWithOutputs(Metrics):
    '''
    A useful leaker object for when you need to get the total run time for applications. Final result of this leaker is the
    average run time of every app. Run time takes into account the execution time AND transmission time.
    This class also writes everything inside a file.
    '''
    def __init__(self, result_path:str):
        self.time = {}
        self.counter = {}
        self.result = None

        self.__app_log_file = open(f'{result_path}_apps.csv', 'w')
        self.__app_log_writer = csv.DictWriter(self.__app_log_file, YAFS_APP_LOG_HEADERS)
        self.__app_log_writer.writeheader()

        self.__link_log_file = open(f'{result_path}_links.csv', 'w')
        self.__link_log_writer = csv.DictWriter(self.__link_log_file, YAFS_LINK_LOG_HEADERS)
        self.__link_log_writer.writeheader()


    def flush(self):
        self.__app_log_file.flush()
        self.__link_log_file.flush()
    
    def insert(self, value):
        app_name = value['app']
        if app_name in self.time:
            self.time[app_name] += (value['time_out'] - value['time_emit'])
            if value['module.src'] == 'Source_0':
                self.counter[app_name] += 1
        else:
            self.time[app_name] = (value['time_out'] - value['time_emit'])
            self.counter[app_name] = 1
        
        self.__app_log_writer.writerow(value)
        
    def insert_link(self, value):
        self.__link_log_writer.writerow(value)


    def close(self):
        self.__app_log_file.close()
        self.__link_log_file.close()
        times = self.time
        count = self.counter
        stats = [times[name]/count[name] for name in times]
        if len(stats) == 0:
            self.result = NO_RESULT_DEFAULT
            warnings.warn('No Results were captured during simulation. Returning default result.')
        else:
            self.result = statistics.mean(stats)
        


class FilteredMetricLeaker(Metrics):
    '''
    A leaker object that can leak specified information. You can use this to get specific information from YAFS.
    Final result of this object is a tuple of two lists, first one being the application logs and the second one
    being link logs.
    '''
    def __init__(self, app_log_filter:list[str], link_log_filter:list[str]):
        '''
        app_log_filter is a subset of:
          "id", "type", "app", "module", "message", "DES.src", "DES.dst", "TOPO.src", "TOPO.dst", "module.src", "service",
          "time_in","time_out", "time_emit","time_reception"
        
        link_log_filter is a subset of:
          "id", "type", "src", "dst", "app", "latency", "message", "ctime", "size", "buffer"
        '''
        self.app_log = []
        self.link_log = []
        self.app_log_filter = ()
        self.link_log_filter = ()
        if app_log_filter:
            self.app_log_filter = tuple(app_log_filter)
        else:
            raise Exception('At least one item is necessary for app_log_filter.')
        if link_log_filter:
            self.link_log_filter = tuple(link_log_filter)
        else:
            raise Exception('At least one item is necessary for link_log_filter.')

    def flush():
        pass
    
    def insert(self, value):
        self.app_log.append({k: value[k] for k in self.app_log_filter})
        
    def insert_link(self, value):
        self.link_log.append({k: value[k] for k in self.link_log_filter})

    def close(self):
        self.result = (self.app_log, self.link_log)