import sys
import csv
import time
import numpy as np
import
np.random.seed(42)
import random
import pandas as pd
import logging
import copy
import math
import cProfile, pstats, io
from pstats import SortKey
from queue import Queue


class VRP_Problem:
    def __init__(self, number_of_vehicles):
        self.number_of_vehicles = number_of_vehicles
        self.number_of_jobs = None
        self.job_locations = []
        self.start_times = []
        self.end_times = []
        self.service_times = []
        self.memo = {}
        self.prefix_memo = {}
        self.prefix_labels = []
        self.prefix_label_times = None
        self.prefix_label_dist = None
        self.queue = Queue(maxsize=0)
        self.test3_restricted_labels = {}
        self.distances_array = []
        self.travel_times_array = []
        self.LDT_array = []
        self.before_sets = []
        self.optimal_cost = None
        self.optimal_path = None
        self.t = None
        self.run_time = None
        self.df = None
        self.df1 = None
        self.prev_visited = []
        self.new_visited = []
        if self.number_of_vehicles == 1:
            self.prev_last_point = None
            self.prev_time = 0
            self.prev_dist = 0
            self.new_last_point = None
            self.new_time = None
            self.new_dist = None
        else:
            self.prev_last_point = [0 for i in range(self.number_of_vehicles)]  # 0 is the depot location
            self.prev_time = [0 for i in range(self.number_of_vehicles)]
            self.prev_dist = [0 for i in range(self.number_of_vehicles)]
            self.new_last_point = [0 for i in range(self.number_of_vehicles)]
            self.new_time = [0 for i in range(self.number_of_vehicles)]
            self.new_dist = [0 for i in range(self.number_of_vehicles)]
            self.key_version = 0
            self.prev_key_version = 0
        # self.first = None
        self.first = [0 for i in range(self.number_of_vehicles)]
        self.all_points_set = []
        self.dumas_before_sets = []
        self.VRP_before_set = None
        self.TW = [False for i in range(self.number_of_vehicles)]
        self.vehicle_order = [i for i in range(self.number_of_vehicles)]
        self.added_labels = []
        self.deleted_labels = []
        self.label_check = {}
        self.dup_lab_rejected = 0
        self.TW_rejected = 0
        self.test1_rejected = 0
        self.test2_rejected = 0
        self.dom_lab_rejected = 0
        self.jump_ahead_rejected = 0
        self.sorted_nlp = None
        self.sorted_nt = [0 for i in range(self.number_of_vehicles)]
        self.sorted_nd = [0 for i in range(self.number_of_vehicles)]
        self.sorted_vo = [i for i in range(self.number_of_vehicles)]
        self.rejected_labels = {}
        self.shortcut_memo = {}
        self.labels_considered = 0
        self.b = 0
        self.stopper = False

    # shared functions
    def reset_problem(self):
        self.number_of_jobs = None
        self.number_of_jobs = None
        self.job_locations = []
        self.start_times = []
        self.end_times = []
        self.service_times = []
        self.memo = {}
        self.prefix_memo = {}
        self.prefix_labels = []
        self.prefix_label_times = None
        self.prefix_label_dist = None
        self.queue = Queue(maxsize=0)
        self.test3_restricted_labels = {}
        self.distances_array = []
        self.travel_times_array = []
        self.LDT_array = []
        self.before_sets = []
        self.optimal_cost = None
        self.optimal_path = None
        self.t = None
        self.run_time = None
        self.df = None
        self.df1 = None
        self.prev_visited = []
        self.new_visited = []
        if self.number_of_vehicles == 1:
            self.prev_last_point = None
            self.prev_time = 0
            self.prev_dist = 0
            self.new_last_point = None
            self.new_time = None
            self.new_dist = None
        else:
            self.prev_last_point = [0 for i in range(self.number_of_vehicles)]  # 0 is the depot location
            self.prev_time = [0 for i in range(self.number_of_vehicles)]
            self.prev_dist = [0 for i in range(self.number_of_vehicles)]
            self.prev_key_version = 0
            self.key_version = 0
            self.new_last_point = [0 for i in range(self.number_of_vehicles)]
            self.new_time = [0 for i in range(self.number_of_vehicles)]
            self.new_dist = [0 for i in range(self.number_of_vehicles)]
        # self.first = None
        self.first = [0 for i in range(self.number_of_vehicles)]
        self.all_points_set = []
        self.dumas_before_sets = []
        self.VRP_before_set = None
        self.TW = [False for i in range(self.number_of_vehicles)]
        self.vehicle_order = [i for i in range(self.number_of_vehicles)]
        self.added_labels = []
        self.deleted_labels = []
        self.label_check = {}
        self.dup_lab_rejected = 0
        self.TW_rejected = 0
        self.test1_rejected = 0
        self.test2_rejected = 0
        self.dom_lab_rejected = 0
        self.jump_ahead_rejected = 0
        self.sorted_nlp = None
        self.sorted_nt = [0 for i in range(self.number_of_vehicles)]
        self.sorted_nd = [0 for i in range(self.number_of_vehicles)]
        self.sorted_vo = [i for i in range(self.number_of_vehicles)]
        self.rejected_labels = {}
        self.shortcut_memo = {}
        self.labels_considered = 0
        self.b = 0
        self.stopper = False
        return

    def read_in_data(self, data, travel_times_multiplier):
        x = []
        y = []
        for d in csv.DictReader(open(data), delimiter='\t'):
            x.append(int(d['xcoord']))
            y.append(int(d['ycoord']))
            self.job_locations.append((int(d['xcoord']), int(d['ycoord'])))
            self.start_times.append(int(d['start']))
            self.end_times.append(int(d['end']))
            self.service_times.append(int(d['service']))
        self.job_locations = np.array(self.job_locations)
        self.number_of_jobs = len(self.start_times)
        self.all_points_set = list(range(self.number_of_jobs))
        self.distances_array = np.array([[np.linalg.norm(self.job_locations[i] - self.job_locations[j])
                                          for i in range(self.number_of_jobs)]
                                         for j in range(self.number_of_jobs)])
        self.distances_array = np.round(self.distances_array, 2)
        self.travel_times_array = travel_times_multiplier * np.round(self.distances_array, 2)
        self.df = pd.DataFrame({'xcoord': x, 'ycoord': y, 'start time': self.start_times, 'end time': self.end_times,
                                'service time': self.service_times})
        logging.info(self.df)
        logging.info(self.distances_array)
        return

    def random_data_generator(self, instances, timeframe, locationframe, servicetime, serviceframe,
                              travel_times_multiplier, save_name):
        if self.number_of_vehicles == 1:
            self.number_of_jobs = instances
            name = range(0, instances)
            coordinates = range(-locationframe, locationframe)
            x = random.choices(coordinates, weights=None, k=instances)
            y = random.choices(coordinates, weights=None, k=instances)
            for i in name: self.job_locations.append((x[i], y[i]))
            self.job_locations = np.array(self.job_locations)
            time0 = range(0, timeframe)
            start = random.choices(time0, weights=None, k=instances)
            for i in name: self.start_times.append(start[i])
            end = []
            for i in name: end.append(random.randrange(start[i]+25,timeframe+25))
            #for i in name: end.append(random.randrange(start[i] + 25, start[i] + 25))
            for i in name: self.end_times.append(end[i])
            if servicetime == True:
                service_time = random.choices(range(0, serviceframe), weights=None, k=instances)
            else:
                service_time = [0] * self.number_of_jobs
            for i in name: self.service_times.append(service_time[i])
            self.distances_array = np.array([[np.linalg.norm(self.job_locations[i] - self.job_locations[j])
                                              for i in range(self.number_of_jobs)]
                                             for j in range(self.number_of_jobs)])
            self.distances_array = np.round(self.distances_array, 2)
            self.travel_times_array = travel_times_multiplier * np.round(self.distances_array, 2)
            self.all_points_set = list(range(self.number_of_jobs))
            testdata = pd.DataFrame(
                {'name': name, 'xcoord': x, 'ycoord': y, 'start': start, 'end': end, 'service': service_time})
            testdata.to_csv('testinstances.csv', sep='\t', index=False)
            self.df = pd.DataFrame(
                {'xcoord': x, 'ycoord': y, 'start time': start, 'end time': end, 'service time': service_time})
        else:
            self.number_of_jobs = instances
            name = range(0, instances)
            coordinates = range(-locationframe, locationframe)
            x = [0] + random.choices(coordinates, weights=None, k=instances - 1)
            y = [0] + random.choices(coordinates, weights=None, k=instances - 1)
            for i in name: self.job_locations.append((x[i], y[i]))
            self.job_locations = np.array(self.job_locations)
            time0 = range(0, timeframe)
            start = random.choices(time0, weights=None, k=instances - 1)
            end = []
            #for i in range(len(start)): end.append(random.randrange(start[i] + 25, timeframe + 25))
            for i in range(len(start)): end.append(random.randrange(start[i] + 25, start[i]+225))
            start = [0] + start
            end = [timeframe] + end
            for i in name: self.start_times.append(start[i])
            for i in name: self.end_times.append(end[i])
            if servicetime == True:
                service_time = [0] + random.choices(range(0, serviceframe), weights=None, k=instances)
            else:
                service_time = [0] * self.number_of_jobs
            for i in name: self.service_times.append(service_time[i])
            self.distances_array = np.array([[np.linalg.norm(self.job_locations[i] - self.job_locations[j])
                                              for i in range(self.number_of_jobs)]
                                             for j in range(self.number_of_jobs)])
            self.distances_array = np.round(self.distances_array, 2)
            self.travel_times_array = np.round(self.distances_array, 2)
            self.all_points_set = list(range(self.number_of_jobs))
            testdata = pd.DataFrame(
                {'name': name, 'xcoord': x, 'ycoord': y, 'start': self.start_times, 'end': self.end_times,
                 'service': self.service_times})
            testdata.to_csv(save_name, sep='\t', index=False)
            print(testdata)

            self.df = pd.DataFrame(
                {'xcoord': x, 'ycoord': y, 'start time': self.start_times, 'end time': self.end_times,
                 'service time': self.service_times})
            print(self.df)


        return

a = VRP_Problem(number_of_vehicles=2)
original_stdout = sys.stdout # Save a reference to the original standard output

with open('log_file.txt', 'a') as f:
    sys.stdout = f # Change the standard output to the file we created.
    a.random_data_generator(instances=7, timeframe=2000, locationframe=100, servicetime=True, serviceframe=25,
                            travel_times_multiplier=1, save_name='log_test')
    sys.stdout = original_stdout # Reset the standard output to its original value

