import csv
import time
import numpy as np
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
        self.queue = Queue(maxsize = 0)
        self.test3_restricted_labels = {}
        self.distances_array = []
        self.travel_times_array = []
        self.LDT_array = []
        self.before_sets = []
        self.optimal_cost=None 
        self.optimal_path=None
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
            self.prev_last_point = [0 for i in range(self.number_of_vehicles)] #0 is the depot location
            self.prev_time = [0 for i in range(self.number_of_vehicles)]
            self.prev_dist = [0 for i in range(self.number_of_vehicles)]
            self.new_last_point = [0 for i in range(self.number_of_vehicles)]
            self.new_time = [0 for i in range(self.number_of_vehicles)]
            self.new_dist = [0 for i in range(self.number_of_vehicles)]
            self.key_version = 0
            self.prev_key_version = 0
        #self.first = None
        self.first = [0 for i in range(self.number_of_vehicles)]
        self.all_points_set = []
        self.dumas_before_sets = []
        self.VRP_before_set = None
        self.TW = [False for i in range(self.number_of_vehicles)]
        self.vehicle_order = [i for i in range(self.number_of_vehicles)]
        self.added_labels = []
        self.deleted_labels =[]
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

#shared functions
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
        self.queue = Queue(maxsize = 0)
        self.test3_restricted_labels = {}
        self.distances_array = []
        self.travel_times_array = []
        self.LDT_array = []
        self.before_sets = []
        self.optimal_cost=None 
        self.optimal_path=None
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
            self.prev_last_point = [0 for i in range(self.number_of_vehicles)] #0 is the depot location
            self.prev_time = [0 for i in range(self.number_of_vehicles)]
            self.prev_dist = [0 for i in range(self.number_of_vehicles)]
            self.prev_key_version = 0
            self.key_version = 0
            self.new_last_point = [0 for i in range(self.number_of_vehicles)]
            self.new_time = [0 for i in range(self.number_of_vehicles)]
            self.new_dist = [0 for i in range(self.number_of_vehicles)]
        #self.first = None
        self.first = [0 for i in range(self.number_of_vehicles)]
        self.all_points_set = []
        self.dumas_before_sets = []
        self.VRP_before_set = None
        self.TW = [False for i in range(self.number_of_vehicles)]
        self.vehicle_order = [i for i in range(self.number_of_vehicles)]
        self.added_labels = []
        self.deleted_labels =[]
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
            self.job_locations.append((int(d['xcoord']),int(d['ycoord'])))
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
        self.travel_times_array = travel_times_multiplier*np.round(self.distances_array, 2)
        self.df = pd.DataFrame({'xcoord' : x, 'ycoord' : y, 'start time' : self.start_times, 'end time' : self.end_times, 'service time' : self.service_times})
        logging.info(self.df)
        logging.info(self.distances_array)
        return

    def random_data_generator(self, instances, timeframe, locationframe, servicetime, serviceframe, travel_times_multiplier, save_name):
        if self.number_of_vehicles == 1:
            self.number_of_jobs = instances
            name = range(0,instances)
            coordinates = range(-locationframe,locationframe)
            x = random.choices(coordinates, weights=None, k=instances)
            y = random.choices(coordinates, weights=None, k=instances)
            for i in name: self.job_locations.append((x[i],y[i]))
            self.job_locations = np.array(self.job_locations)
            time0 = range(0,timeframe)
            start = random.choices(time0, weights=None, k=instances)
            for i in name: self.start_times.append(start[i])
            end=[]
            for i in name: end.append(random.randrange(start[i]+25,timeframe+25))
            for i in name: self.end_times.append(end[i])
            if servicetime == True:
                service_time = random.choices(range(0,serviceframe), weights=None, k=instances)
            else:
                service_time=[0]*self.number_of_jobs
            for i in name: self.service_times.append(service_time[i])
            self.distances_array = np.array([[np.linalg.norm(self.job_locations[i] - self.job_locations[j])
                                 for i in range(self.number_of_jobs)]
                                for j in range(self.number_of_jobs)])
            self.distances_array = np.round(self.distances_array, 2)
            self.travel_times_array = travel_times_multiplier * np.round(self.distances_array, 2)
            self.all_points_set = list(range(self.number_of_jobs))
            testdata = pd.DataFrame({'name' : name, 'xcoord' : x, 'ycoord' : y, 'start' : start, 'end' : end, 'service' : service_time})
            testdata.to_csv('testinstances.csv', sep='\t',index=False)
            self.df = pd.DataFrame({'xcoord' : x, 'ycoord' : y, 'start time' : start, 'end time' : end, 'service time' : service_time})
        else:
            self.number_of_jobs = instances
            name = range(0,instances)
            coordinates = range(-locationframe,locationframe)
            x = [0] + random.choices(coordinates, weights=None, k=instances-1)
            y = [0] + random.choices(coordinates, weights=None, k=instances-1)
            for i in name: self.job_locations.append((x[i],y[i]))
            self.job_locations = np.array(self.job_locations)
            time0 = range(0,timeframe)
            start = random.choices(time0, weights=None, k=instances-1)
            end=[]
            for i in range(len(start)): end.append(random.randrange(start[i]+25,timeframe+25))
            start = [0] + start
            end = [timeframe] + end
            for i in name: self.start_times.append(start[i])
            for i in name: self.end_times.append(end[i])
            if servicetime == True:
                service_time = [0] + random.choices(range(0,serviceframe), weights=None, k=instances)
            else:
                service_time=[0]*self.number_of_jobs
            for i in name: self.service_times.append(service_time[i])
            self.distances_array = np.array([[np.linalg.norm(self.job_locations[i] - self.job_locations[j])
                                 for i in range(self.number_of_jobs)]
                                for j in range(self.number_of_jobs)])
            self.distances_array = np.round(self.distances_array, 2)
            self.travel_times_array = np.round(self.distances_array, 2)
            self.all_points_set = list(range(self.number_of_jobs))
            testdata = pd.DataFrame({'name' : name, 'xcoord' : x, 'ycoord' : y, 'start' : self.start_times, 'end' : self.end_times, 'service' : self.service_times})
            testdata.to_csv(save_name, sep='\t',index=False)
            self.df = pd.DataFrame({'xcoord' : x, 'ycoord' : y, 'start time' : self.start_times, 'end time' : self.end_times, 'service time' : self.service_times})

        return 

    def dumas_latest_departure_time(self, x, y):
        ldt = self.end_times[y]-self.service_times[y]-self.travel_times_array[x][y]-self.service_times[x]
        return ldt

    def special_values(self):
        self.dumas_before_sets = [[ ] for y in self.all_points_set]
        rows, cols = (range(len(self.distances_array)), range(len(self.distances_array)))
        self.LDT_array = np.array([[self.dumas_latest_departure_time(i,j) for i in rows] for j in cols])
        logging.info(f"LDT array = {self.LDT_array}")
        self.before_sets = [self.dumas_before(i)[i] for i in self.all_points_set]
        logging.info(f"before sets = {self.before_sets}")
        return

    def dumas_before(self, x):
        for i in self.all_points_set:
                if self.start_times[i]+self.service_times[i]+self.travel_times_array[i][x] > self.end_times[x]:
                    self.dumas_before_sets[i].append(x)
                    logging.debug(self.dumas_before_sets)
                else:
                    continue
        return self.dumas_before_sets

#TSPTW functions
    def restricted_labels_check(self):
        logging.info(f"checking if ({self.new_visited}, {self.new_last_point}, {self.new_time}) is on restricted list")
        if self.test3_restricted_labels.get((tuple(sorted(self.new_visited)), self.new_last_point)) == None:
            logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) IS NOT on restricted list")
            test3 = True
        else:
            if self.new_time < self.test3_restricted_labels.get((tuple(sorted(self.new_visited)), self.new_last_point)):
                logging.debug(f"{self.new_time} < {self.test3_restricted_labels.get((tuple(sorted(self.new_visited)), self.new_last_point))}")
                logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) IS NOT on restricted list")
                test3 = True
            else:
                logging.debug(f"{self.new_time} >= {self.test3_restricted_labels.get((tuple(sorted(self.new_visited)), self.new_last_point))}")
                logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) IS on restricted list")
                test3 = False    
        return test3

    def time_window_check(self):
        logging.info('time window check')
        if self.new_time > self.end_times[self.new_last_point]:
            TW = False
            logging.info(f"TW infeasible, {self.new_time} > {self.end_times[self.new_last_point]}")
        else: 
            TW = True
            logging.info(f"TW feasible, {self.new_time} <= {self.end_times[self.new_last_point]}")
        return TW

    def dumas_test2(self):
        logging.debug(f"before({self.new_last_point})={self.before_sets[self.new_last_point]}")
        logging.debug(f"S = {self.new_visited}")
        flag = 0
        if all(x in self.before_sets[self.new_last_point] for x in self.new_visited):
            flag = 1
        if flag:
            test2 = True
            logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) passes test 2")
        else: 
            test2 = False
            logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) fails test 2")
            
        return test2

    def dumas_first(self):
        res1 = {k: v for k, v in self.memo.items() if k[0]==self.new_visited and k[1]==self.new_last_point}        
        res2 = {value: key for key, value in res1.items()}        
        if len(res2) == 0: 
            self.first = self.new_time
        else:
            res3 = min(res2.keys(), key=lambda x: res2[x][2])            
            self.first = res2[res3][2]            
        return

    def dumas_test1(self):
        check = [i for i in all_points_set if i not in self.new_visited]
        if len(check) == 0:
            test1 = True
        elif self.first > min(self.LDT_array[j][self.new_last_point] for j in check): 
            logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) fails test 1")
            test1 = False
        else:
            logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) passes test 1")
            test1 = True
        return test1

    def dumas_test3(self):
        logging.debug(f"adding to restricted labels")
        check1 = [i for i in all_points_set if i not in self.new_visited]
        for x in check1:
                new_visited1 = sorted(self.new_visited)
                new_last_point1 = self.new_last_point
                new_time1 = self.new_time
                if self.new_time <= self.LDT_array[x][self.new_last_point]:
                    new_visited1 = [x] + list(new_visited1)
                    new_last_point1 = x
                    new_time1 = self.new_time+self.service_times[self.new_last_point]+self.travel_times_array[self.new_last_point][x]
                    check2 = [i for i in all_points_set if i not in new_visited1]
                    new_visited1 = tuple(sorted(new_visited1))
                    for y in check2:
                        new_visited2 = list(sorted(new_visited1))
                        new_last_point2 = new_last_point1
                        new_time2 = new_time1
                        if new_time1 > self.LDT_array[y][new_last_point1]:
                            new_visited2 += [y]
                            new_visited2 = tuple(sorted(new_visited2))
                            new_last_point2 = y
                            new_time2 = new_time1+self.service_times[new_last_point1]+self.travel_times_array[new_last_point1][y]
                            if self.test3_restricted_labels.get((new_visited2, new_last_point2)) == None:
                                self.test3_restricted_labels[(new_visited1, new_last_point1)] = new_time1
                            elif new_time2 < self.test3_restricted_labels[(new_visited2, new_last_point2)]:
                                self.test3_restricted_labels[(new_visited1, new_last_point1)] = new_time1
                            else:
                                continue                        
                        else: 
                            continue                        
                else: 
                    continue
        logging.debug(f"new restricted labels: {self.test3_restricted_labels}")
        return    

    def dominance_test(self):
        logging.info(f"starting dominance check:")
        if len({k: v for k, v in self.memo.items() if k[0]==self.new_visited and k[1]==self.new_last_point}) != 0: #possible dominated situation            
            if (self.new_time <= self.first) and (self.new_dist <= self.memo[(self.new_visited, self.new_last_point, self.first)][0]): #time and cost improvement, replace label
                del self.memo[(self.new_visited, self.new_last_point, self.first)]
                self.memo[(self.new_visited, self.new_last_point, self.new_time)] = (self.new_dist, self.prev_last_point, self.prev_time)
                #self.queue.remove((self.new_visited, self.new_last_point, self.first))
                self.queue.put((self.new_visited, self.new_last_point, self.new_time))
                logging.info(f"label ({self.new_visited}, {self.new_last_point}, {self.new_time}) dominated, case 1, replaces label ({self.new_visited},{self.new_last_point},{self.first})")
            elif (self.new_time == self.first) and (self.new_dist < self.memo[(self.new_visited, self.new_last_point, self.first)][0]): #same time, cost improvement, replace old label with new
                self.memo[(self.new_visited, self.new_last_point, self.first)] = (self.new_dist, self.prev_last_point, self.prev_time)
                self.queue.put((self.new_visited, self.new_last_point, self.new_time))
                logging.info(f"label ({self.new_visited}, {self.new_last_point}, {self.new_time}) dominated, case 2, same time, better distance, updates old label with new distance")
            elif (self.new_time < self.first) and (self.new_dist >= self.memo[(self.new_visited, self.new_last_point, self.first)][0]): #time improvement only, add new label
                self.memo[(self.new_visited, self.new_last_point, self.new_time)] = (self.new_dist, self.prev_last_point, self.prev_time)
                self.queue.put((self.new_visited, self.new_last_point, self.new_time))
                logging.info(f"label ({self.new_visited}, {self.new_last_point}, {self.new_time}) dominated, case 3, better time, worse cost, adds label")
            elif (self.new_time >= self.first) and (self.new_dist  < self.memo[(self.new_visited, self.new_last_point, self.first)][0]): #cost improvement only, add new label
                self.memo[(self.new_visited, self.new_last_point, self.new_time)] = (self.new_dist, self.prev_last_point, self.prev_time)
                self.queue.put((self.new_visited, self.new_last_point, self.new_time))
                logging.info(f"label ({self.new_visited}, {self.new_last_point}, {self.new_time}) dominated, case 4, slower time, better cost, adds label")
            else:
                logging.info(f"label ({self.new_visited}, {self.new_last_point}, {self.new_time}) dominated, case 5, no label created")
                #if (new_time >= first) and (new_dist >= memo[(new_visited, new_last_point, first)][0]): #no improvement, no label added
                #queue += [(new_visited, new_last_point, new_time)]                 
        else: #len({k: v for k, v in memo.items() if k[0]==new_visited and k[1]==new_last_point}) == 0: #new label
            self.memo[(self.new_visited, self.new_last_point, self.new_time)] = (self.new_dist, self.prev_last_point, self.prev_time)
            self.queue.put((self.new_visited, self.new_last_point, self.new_time))
            logging.info(f"no (S,i) label exists, {self.new_visited, self.new_last_point, self.new_time} added") 
        logging.debug(f"the queue is {self.queue}")
        logging.debug(f"the memo is {self.memo}")
        return

    def retrace_optimal_path_TSPTW(self, memo: dict, n: int) -> [[int], float]:
        points_to_retrace = tuple(range(n))
    
        full_path_memo = dict((k, v) for k, v in memo.items() if k[0] == points_to_retrace)
        if len(full_path_memo) == 0: 
            optimal_cost=None, 
            optimal_path=None, 
            df=None
        elif len(full_path_memo) !=0:
            path_key = min(full_path_memo.keys(), key=lambda x: full_path_memo[x][0])
            end_time = path_key[2]
            optimal_path_arrival_times=[end_time]
            last_point = path_key[1]
            optimal_cost, next_to_last_point, prev_time = memo[path_key]
            optimal_path = [last_point]
            res1 = [i for i in points_to_retrace if i not in last_point]
            points_to_retrace = tuple(sorted(res1))

            while next_to_last_point is not None:
                last_point = next_to_last_point
                end_time = prev_time
                path_key = (points_to_retrace, last_point, prev_time)
                _, next_to_last_point, prev_time = memo[path_key]
                optimal_path = [last_point] + optimal_path
                optimal_path_arrival_times = [end_time] + optimal_path_arrival_times
                res1 = [i for i in points_to_retrace if i not in last_point]
                points_to_retrace = tuple(sorted(res1))

            start1 = [self.start_times[i] for i in optimal_path]
            end1 = [self.end_times[i] for i in optimal_path]
            optimal_path_departure_times = [max(optimal_path_arrival_times[i]+self.service_times[i], start1[i]) for i in range(len(self.distances_array))]
            self.optimal_path = optimal_path
            self.optimal_cost = optimal_cost
            self.df1 = pd.DataFrame({'opt path': optimal_path, 'start': start1, 'arrival': optimal_path_arrival_times, 'departure': optimal_path_departure_times, 'end': end1 })
            print("time check:")
            print(self.df1)
        return optimal_path, optimal_cost
    
    def Dumas_TSPTW_Solve(self, T1, T2, T3):
        self.dumas_before_sets = [[ ] for y in self.all_points_set]
        self.special_values()
        
        self.memo = {(tuple([i]), i, 0): tuple([0, None, 0]) for i in range(self.number_of_jobs)} 
        for i in range(self.number_of_jobs):
            self.queue.put((tuple([i]), i, 0))
        while self.queue.qsize() != 0: 
            self.prev_visited, self.prev_last_point, self.prev_time = self.queue.get()
            while self.memo.get((self.prev_visited, self.prev_last_point, self.prev_time)) == None:
                self.prev_visited, self.prev_last_point, self.prev_time = self.queue.get()
            self.prev_dist, _, _ = self.memo[(self.prev_visited, self.prev_last_point, self.prev_time)]
            logging.info(f"extending from ({self.prev_visited}, {self.prev_last_point}, {self.prev_time})")
            to_visit = [i for i in all_points_set if i not in self.prev_visited]
            
            for self.new_last_point in to_visit:
                self.new_visited = tuple(sorted(list(self.prev_visited) + [self.new_last_point]))
                self.new_dist = self.prev_dist + self.distances_array[self.prev_last_point][self.new_last_point]
                self.new_time = max(self.prev_time, self.start_times[self.prev_last_point]) + self.service_times[self.prev_last_point] + self.travel_times_array[self.prev_last_point][self.new_last_point]
                logging.info(f"checking ({self.new_visited}, {self.new_last_point}, {self.new_time})")
                if not self.time_window_check():
                    continue #this continue will send you back to the top of the for loop
                logging.info(f"tests started")

                # If test 3 should be executed and label has already been identifed as restricted, continue
                if T3:
                   if not self.restricted_labels_check():
                        logging.info(f"tests ended")
                        continue 
                else: 
                    logging.info(f"test 3 is not being used")
               
                if T2:
                    if not self.dumas_before():
                        self.dumas_test2()
                        logging.info(f"tests ended")
                        continue
                else:
                    logging.info(f"test 2 is not being used")
                
                self.dumas_first()
                if T1:                    
                    if not self.dumas_test1():
                        logging.info(f"tests ended")
                        continue
                else:
                    logging.info(f"test 1 is not being used")
                    
                if T3:
                    self.dumas_test3()

                logging.info(f"tests ended")

                self.dominance_test()

        self.optimal_path, self.optimal_cost = self.retrace_optimal_path_TSPTW(self.memo, self.number_of_jobs)
    
        return self.optimal_path, self.optimal_cost, self.df1

#VRPTW functions
#this checks to see if the exact same label is already in the memo, if so, the current label is abandoned otherwise the other checks proceed
    def duplicate_label_check_VRP_update2(self):
        logging.info(f"starting duplicate label check")
        #first check to see if the current label is stored in the memo or rejected labels dictionaries with a cost equal to the current cost.
        if ((tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version) in self.memo) or ((self.new_visited, self.sorted_nlp, self.sorted_nt, self.key_version) in self.rejected_labels):
            if (tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version) in self.memo:
                costcheck, _, _, _, _ = self.memo[(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version)]
                logging.debug(f"costcheck_memo = {costcheck}")
            elif (self.new_visited, self.sorted_nlp, self.sorted_nt, self.key_version) in self.rejected_labels:
                costcheck = self.rejected_labels[(self.new_visited, self.sorted_nlp, self.sorted_nt, self.key_version)]
                logging.debug(f"costcheck_reject = {costcheck}")
            else:
                costcheck = -1
                logging.debug(f"the current label is not already in the memo or already been rejected")
            
            if tuple(self.sorted_nd) == tuple(costcheck):
                dup_lab_check = False
                logging.info(f"the label {(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version)} is a duplicate")
                self.dup_lab_rejected = self.dup_lab_rejected + 1
            else:
                dup_lab_check = True
        else:
            dup_lab_check = True
            logging.info(f"the label {(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version)} HAS NOT already been encountered")
        logging.info(f"duplicate label check ended.")
        return dup_lab_check 

    def duplicate_label_check_VRP_update3(self):
        #first check to see if the current label is stored in the memo or rejected labels dictionaries with a cost equal to the current cost.
        logging.info(f"starting duplicate label check")
        dup_lab1 = self.label_check
        dup_lab2 = {k: v for k, v in self.rejected_labels.items() if k[0]==self.new_visited and k[1]==self.sorted_nlp}
        if len(dup_lab1)+len(dup_lab2) == 0:
            logging.info(f"the label {(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt))} HAS NOT already been encountered")
        else:
            dup_lab_keys1 = [key for key in dup_lab1.keys()]
            dup_lab_keys2 = [key for key in dup_lab2.keys()]
            dup_lab_keys = dup_lab_keys1 + dup_lab_keys2
            #dup_lab_keys removes the values associated with each dictionary entry in the possibly duplicated labels (dup_lab).
            #what if two label have the same key, but different values?
            logging.debug(f"possible duplicated label keys = {dup_lab_keys}")
            dup_lab_times = [dup_lab_keys[i][2] for i in range(len(dup_lab_keys))]
            #dup_lab_times strips away everything but the times
            logging.debug(f"possible duplicated label times = {dup_lab_times}")
            for i in range(len(dup_lab_values)):
                for j in range(self.number_of_vehicles):
                    dup_lab_times[i][j] = float(round.np(dup_lab_times[i][j],2))
            logging.debug(f"possible duplicated label values = {dup_lab_values}")
            dup_lab_values1 = [value for value in dup_lab1.values()]
            #dup_lab_values removes the keys associated with each memo entry in the possibly duplicate labels (dup_lab1).
            logging.debug(f"possible dominated label values = {dup_lab_values}")
            dup_lab_dist1 = [dup_lab_values1[i][0] for i in range(len(dup_lab_values))]
            dup_lab_dist2 = [value for value in dup_lab2.values()]
            dup_lab_dist = dup_lab_dist1 + dup_lab_dist2
            #dup_lab_dist stips away everything but the costs 
            logging.debug(f"possible dominated label distances = {dup_lab_dist}")
            #time_check 1 and dist_check 1 are set up to record the outcomes of comparing the times and cost (dist) of the current label with the labels in dom_lab to check if the current label is dominated by an existing label
            dup_time_check1 = [[False for j in range(self.number_of_vehicles)] for i in range(len(dup_lab))]
            dup_dist_check1 = [[False for j in range(self.number_of_vehicles)] for i in range(len(dup_lab))]
            logging.debug(f"time check 1 = {dup_time_check1}")
            logging.debug(f"dist check 1 = {dup_dist_check1}")

            for j in range(self.number_of_vehicles):
                for i in range(len(dup_lab)):
                    logging.debug(f"current time = {self.sorted_nt[j]}")
                    logging.debug(f"stored time = {dup_lab_times[i][j]}")
                    if self.sorted_nt[j] == dup_lab_times[i][j]:
                        #compares the jth entry for time in the current label to the jth entry for time in the ith label in the possibly dominated labels
                        dup_time_check1[i][j] = True
                        #records the result in the jth component of ith time_check tuple
                        logging.debug(f"time check 1 = {dup_time_check1}")
                    else:
                        logging.debug(f"time check 1 = {dup_time_check1}")
                        continue
            for j in range(self.number_of_vehicles):
                for i in range(len(dup_lab)):
                    logging.debug(f"current dist = {self.sorted_nd[j]}")
                    logging.debug(f"stored dist = {dup_lab_dist[i][j]}")
                    if self.sorted_nd[j] == dup_lab_dist[i][j]:
                        
                        #compares the jth entry for cost (dist) in the current label to the jth entry for cost (dist) in the ith label in the possibly duplicate labels
                        dist_check1[i][j] = True
                        #records the result in the jth component of ith dist_check tuple
                        logging.debug(f"dist check 1 = {dup_dist_check1}")
                    else:
                        logging.debug(f"dist check 1 = {dup_dist_check1}")
                        continue
            
            logging.debug(f"completed time check 1 = {dup_time_check1}")
            logging.debug(f"completed dist check 1 = {dup_dist_check1}")

            #now, use the results of the time and distance check 1 to determine to see if any stored label is a duplicate of the current label
            dup_check_1_time = [all(time_check1[i]) for i in range(len(dom_lab))]
            logging.debug(f"dom_check_1_time = {dom_check_1_time}")
            #label_dup_curr_by_time processes the results of the previous test to determine if any stored label is a duplicate of the current label.  It is checking if there is any entry in time_check that is [True, True].
            dup_check_1_dist = [all(dist_check1[i]) for i in range(len(dom_lab))]
            logging.debug(f"dom_check_1_dist = {dom_check_1_dist}")
            #label_dup_curr_by_dist processes the results of the previous test to determine if any stored label is a duplicate of the current label.  It is checking if there is any entry in dist_check that is [True, True].
            
            #this loop detects if the current label is completely dominated by any existing label
            dup_check = [False for i in range(len(dom_lab))]
            logging.debug(f"curr_lab_dominated_by_existing_label = {curr_lab_dominated_by_existing_label}")
            for i in range(len(dup_lab)):
                logging.debug(f"dup_check_1_time[{i}] = {dup_check_1_time[i]}")
                logging.debug(f"dup_check_1_dist[{i}] = {dup_check_1_dist[i]}")
                if dup_check_1_time[i] == True and dup_check_1_dist[i] == True:
                    
                    dup_check[i] = True
                    
                else:
                    continue
            if any(dup_check) == True:
                dup_lab_check = False
            else:
                dup_lab_check = True
        return dup_lab_check
        
    def VRP_time_window_check(self):
        logging.info(f"start TW check")  
        for i in range(self.number_of_vehicles):
            logging.debug(f"checking vehcile {i}")
            logging.debug(f"NT = {self.new_time[i]}")
            logging.debug(f"ET = {self.end_times[self.new_last_point[i]]}")
            if self.new_time[i] <= self.end_times[self.new_last_point[i]]:
                self.TW[i] = True
                logging.debug("TW[{i}] = true")                
            else:
                self.TW[i] = False
                logging.debug("TW[{i}] = false")                
        logging.debug(f"TW = {self.TW}")
        if False in self.TW:
            TW_test = False
            self.rejected_labels[(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version)] = list(self.sorted_nd)
        else:
            TW_test = True
        logging.info(f"time window test = {TW_test}")
        return TW_test

    def strong_jump_ahead_check(self):
        ja_check = True
        logging.debug(f"new visited = {self.new_visited}, new last point = {self.sorted_nlp}")
        remaining_jobs = [i for i in self.all_points_set  if i not in self.new_visited]
        logging.debug(f"remaining jobs = {remaining_jobs}")
        if len(remaining_jobs) == 0:
            pass
        else:
            for i in self.sorted_nlp:
                for j in remaining_jobs:
                    logging.debug(f"self.start_times[{i}] = {self.start_times[i]} and self.end_times[{j}] = {self.end_times[j]}")
                    if self.start_times[i]>=self.end_times[j]:
                        ja_check = False
        if ja_check == False:
            self.jump_ahead_rejected = self.jump_ahead_rejected + 1
            logging.info(f"jump ahead check failed")
        else:
            logging.info(f"jump ahead check passed")
        return ja_check

    def before_VRP(self):
        Y = [self.dumas_before_sets[x] for x in self.sorted_nlp]
        self.VRP_before_set = set(Y[0]).intersection(*Y) 
        return self.VRP_before_set

    def VRP_test2(self):
        self.before_VRP()
        logging.debug(f"before({self.sorted_nlp})={self.VRP_before_set}")
        logging.debug(f"S = {self.new_visited}")
        flag = 0
        if(all(x in self.new_visited for x in self.VRP_before_set)):
            flag =1
        if flag:
            test2 = True
            logging.info(f"({self.new_visited}, {self.sorted_nlp}, {self.sorted_nt}) passes test 2")            
        else: 
            test2 = False
            logging.info(f"({self.new_visited}, {self.sorted_nlp}, {self.sorted_nt}) fails test 2")
        if test2 == False:
            self.test2_rejected = self.test2_rejected + 1
            self.rejected_labels[(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version)]=list(self.sorted_nd)
        return test2

    #the first function determines the earliest time a vehicle arrives at the jobs in V_i while servicing the jobs in S_i
    def VRP_first1(self):
        res3 = self.prefix_label_times
        if res3 == None:
            logging.debug(f"current times are earliest times = {self.sorted_nt}")
            for i in range(self.number_of_vehicles):
                self.first[i] = self.sorted_nt[i]
        else:
            for j in range(len(res3)):
                logging.debug(f"key {j} = {res3[j]}")
                for i in range(self.number_of_vehicles):
                    self.first[i] = min(res3[j][i], self.sorted_nt[i])
                    logging.debug(f"first = {self.first}")
        return

    def VRP_test1A(self):
        unreachable_points = [i for i in self.all_points_set if i not in self.new_visited]
        logging.debug(f"unreachable points = {unreachable_points}")
        if len(unreachable_points) != 0:
            self.VRP_first1()
            logging.debug(f"first = {self.first}")
            vehicles_to_check = {i for i in self.vehicle_order}
            logging.debug(f"vehicles to check = {vehicles_to_check}")
            while len(vehicles_to_check) != 0 and len(unreachable_points) != 0:
                vehicle_to_check = vehicles_to_check.pop()
                logging.debug(f"vehicles to check are now = {vehicles_to_check}")
                logging.debug(f"currently checking vehicle = {vehicle_to_check}")
                for j in unreachable_points:
                    if self.first[vehicle_to_check] <= self.LDT_array[j][self.sorted_nlp[vehicle_to_check]]:
                        logging.debug(f"first = {self.first[vehicle_to_check]} <= LDT({j},{self.sorted_nlp[vehicle_to_check]}) = {self.LDT_array[j][self.sorted_nlp[vehicle_to_check]]}")
                        unreachable_points = [i for i in unreachable_points if i not in [j]]
                        logging.debug(f"unreachable points are now = {unreachable_points}")
                    else:
                        logging.debug(f"first = {self.first[vehicle_to_check]} > LDT({j},{self.sorted_nlp[vehicle_to_check]}) = {self.LDT_array[j][self.sorted_nlp[vehicle_to_check]]}")
                logging.debug(f"length of unreachable points = {len(unreachable_points)}")
                logging.debug(f"length of vehicles to check = {len(vehicles_to_check)}")
         
            if len(unreachable_points) == 0:
                test1 = True
                logging.info(f"test 1 passed")
            else:
                test1 = False
                logging.info(f"test 1 failed")
        else:
            test1 = True
            logging.info(f"test 1 passed")
        if test1 == False:
            self.test1_rejected = self.test1_rejected + 1
            self.rejected_labels[(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version)]=list(self.sorted_nd)
        return test1 

    #dominance test checks to see if the current label has is totally dominated by any label already stored.  If so the current label is not added.  Otherwise the current label is added and then it checks to see if the current label totally dominates any existing label.  If so those existing labels that are totally dominated are deleted.  If no existing labels are dominated nothing happens.
    def VRP_dominance_test_update4(self):
        #we say that one label dominates another if each vehicle in a label arrives at the desired location at least as fast and at least as cheap as another label
        #this test checks to see if the current label dominates any existing labels or if it dominates any current labels.
        logging.info(f"starting dominance check:")
        dom_lab = self.prefix_labels
        #dom_lab = self.label_check = {k: v for k, v in self.memo.items() if k[0]==self.new_visited and k[1]==self.sorted_nlp}.  
        #These are the labels that could possibly dominate the current label or be dominated by the current label.
        logging.debug(f"possible dominated labels = {dom_lab}")
        if dom_lab == None: #no labels exist to dominate the current label
            logging.info(f"no (S,V_i) label exists, {self.new_visited, self.sorted_nlp, self.sorted_nt, self.key_version} added") 
            self.memo[(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version)] = (self.sorted_nd, self.prev_last_point, self.prev_time, self.sorted_vo, self.prev_key_version)
            self.queue.put((tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version))
            self.prefix_memo[(tuple(self.new_visited), tuple(self.sorted_nlp))] = [(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version)]
            logging.debug(f"prefix_memo = {self.prefix_memo}")
           
        else:    #possible dominated situation            
            dom_check_1_time = [None for i in range(len(dom_lab))]
            dom_check_2_time = [None for i in range(len(dom_lab))]
            dom_check_3_time = [None for i in range(len(dom_lab))]
            logging.debug(f"dom_check_1_time = {dom_check_1_time}")
            #dom_check_#_dist checks to see if there is an existing label that dominates the distance for each vehicle for the current label
            dom_check_1_dist = [None for i in range(len(dom_lab))]
            dom_check_2_dist = [None for i in range(len(dom_lab))]
            for i in range(len(dom_lab)):
                dom_check_1_time[i] = all(x >= y for x, y in zip(self.sorted_nt, self.prefix_label_times[i]))
                dom_check_1_dist[i] = all(x >= y for x, y in zip(self.sorted_nd, self.prefix_label_dist[i]))
                dom_check_2_time[i] = all(x < y for x, y in zip(self.sorted_nt, self.prefix_label_times[i]))
                dom_check_2_dist[i] = all(x <= y for x, y in zip(self.sorted_nd, self.prefix_label_dist[i]))                       
            #dom_check_1 checks to see if there is an existing label that dominates both times and distances for the current label.    
            logging.debug(f"dom_check_1_time = {dom_check_1_time}, dom_check_1_dist = {dom_check_1_dist}, dom_check_2_time = {dom_check_2_time}, dom_check_2_dist= {dom_check_2_dist}")
            dom_check_1 = False
            for i in range(len(dom_lab)):
                if dom_check_1_time[i] and dom_check_1_dist[i]:
                    dom_check_1 = True
                    dominator = i
                else:
                    continue
            logging.debug(f"dom_check_1 = {dom_check_1}")
            #dom_check_2 checks to see if the current label dominates both time and distance for an existing label.    
            dom_check_2 = False
            dominated = []
            for i in range(len(dom_lab)):
                if dom_check_2_time[i] and dom_check_2_dist[i]:
                    dom_check_2 = True
                    dominated = dominated + [self.prefix_labels[i]]
                else:
                    continue
            logging.debug(f"dom_check_2 = {dom_check_2}")
            logging.debug(f"dominated = {dominated}")
            #no labels are added or deleted if dom_check_1 is true because an existing label dominates the current one
            if dom_check_1:
                logging.info(f"the current label ({self.new_visited, self.sorted_nlp, self.sorted_nt}) with costs {self.sorted_nd} is dominated by the existing label ({self.new_visited, self.sorted_nlp, self.prefix_label_times[dominator]}) with costs {self.prefix_label_dist[dominator]} so no new label is added.")
            #existing labels are deleted if dom_check_1 is true because the current label dominates those labels, the current label will be added later
            elif dom_check_2:
                if len(dominated) != 0:
                    temp = self.prefix_memo[(tuple(self.new_visited), tuple(self.sorted_nlp))]
                    logging.debug(f"temp = {temp}")
                    
                    for i in dominated:
                        logging.info(f"the current label ({self.new_visited, self.sorted_nlp, self.sorted_nt}) with costs {self.sorted_nd} dominates the existing label {i} with costs {self.memo[i][0]} so the existing label is deleted.")
                        del self.memo[i]
                        #self.queue.remove(i)
                        temp.remove(i)
                    logging.debug(f"temp = {temp}")
                    self.prefix_memo[(tuple(self.new_visited), tuple(self.sorted_nlp))] = temp
                    logging.debug(f"prefix_memo = {self.prefix_memo}")
                    
                else:
                    logging.info(f"the current label ({self.new_visited, self.sorted_nlp, self.sorted_nt, self.key_version}) with costs {self.sorted_nd} DOES NOT dominate the existing label ({self.new_visited, self.sorted_nlp, self.prefix_label_times[i]}) with costs {self.prefix_label_dist[i]} so the existing label is not deleted.")
            #all other situations are 'incomplete dominance' so no labels are deleted and the current label will be added later.
            #as long as the current label is not dominated it needs to be added, but this must be done carefull so as not to replace a non-dominated existing label
            if not dom_check_1:
                while (tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version) in dom_lab:
                    self.key_version = self.key_version + 1                   
                    logging.debug(f"key_version = {self.key_version}")
                                
                self.memo[(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version)] = (self.sorted_nd, self.prev_last_point, self.prev_time, self.sorted_vo, self.prev_key_version)
                self.queue.put((tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version))
                if (tuple(self.new_visited), tuple(self.sorted_nlp)) in self.prefix_memo:
                    temp = self.prefix_memo[(tuple(self.new_visited), tuple(self.sorted_nlp))]
                    logging.debug(f"temp = {temp}")
                    temp.append((tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version))
                    logging.debug(f"temp = {temp}")
                    self.prefix_memo[(tuple(self.new_visited), tuple(self.sorted_nlp))] = temp
                    logging.debug(f"prefix_memo = {self.prefix_memo}")                    
                else:
                    self.prefix_memo[(tuple(self.new_visited), tuple(self.sorted_nlp))] = [(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version)]
                    logging.debug(f"prefix_memo = {self.prefix_memo}")
                logging.info(f"the current label ({self.new_visited, self.sorted_nlp, self.sorted_nt, self.key_version}) with costs {self.sorted_nd} is a duplicate label to some existing label, but with different values. So the current label is updated and added.")
        return
       
    def prefix_search2(self):
        self.label_check.clear()
        self.prefix_labels = self.prefix_memo.get((tuple(self.new_visited), tuple(self.sorted_nlp)))
        if self.prefix_labels != None:
            for i in self.prefix_labels:
                self.label_check[i] = self.memo[i]
        else:
            self.label_check = {}
        logging.debug(self.label_check)
        
        if self.prefix_labels != None:
            self.prefix_label_times = [self.prefix_labels[i][2] for i in range(len(self.prefix_labels))]
            self.prefix_label_dist = []
            for i in self.prefix_labels:
                self.prefix_label_dist = self.prefix_label_dist + [self.memo[i][0]]
        else:
            self.prefix_label_times = None 
            self.prefix_label_dist = None 
        logging.debug(f"prefix time = {self.prefix_label_times}, prefix dist = {self.prefix_label_dist}, prefix labels = {self.prefix_labels}")
        return    
            
    def retrace_optimal_path_VRP(self, memo: dict, n: int) -> [[int], float]:
        if self.stopper == True:
            print(f"timed out no solution reached")
            print(f"the time is {round(time.time() - self.t,3)}")
            print(f"the memo length is {len(self.memo)}")
        else:
            points_to_retrace = tuple(range(self.number_of_jobs))    
            full_path_memo = dict((k, v) for k, v in self.memo.items() if k[0] == points_to_retrace)
        
            logging.debug(f"full path meemo = {full_path_memo}")
            for x in full_path_memo:
                logging.debug(f"x = {x}")
                current_cost_vector = list(full_path_memo[x][0])
                logging.debug(f"current cost vector = {current_cost_vector}")
                for j in range(self.number_of_vehicles):
                    logging.debug(f"j = {j}")
                    logging.debug(f"current vehicle locations = {x[1]}")
                    logging.debug(f"current cost vector = {full_path_memo[x][0]}")
                    logging.debug(f"current location for vehicle {j} = {x[1][j]}")
                    current_cost_vector[j] = full_path_memo.get(x)[0][j]+self.distances_array[x[1][j]][0]
                    logging.debug(f"updated distances = {current_cost_vector}")
                    full_path_memo[x] = (tuple(current_cost_vector), full_path_memo.get(x)[1], full_path_memo.get(x)[2], full_path_memo.get(x)[3], full_path_memo.get(x)[4])
                    logging.debug(f"full path memo[{x}] = {full_path_memo[x]}")
            logging.debug(f"updated cost full path memo = {full_path_memo}")     

            if len(full_path_memo) == 0: 
                optimal_cost=None, 
                optimal_path=None, 
                self.df1=None
            elif len(full_path_memo) !=0:
                path_key = min(full_path_memo.keys(), key=lambda x: sum(full_path_memo[x][0])) 
                logging.debug(f"path key = {path_key}")
                last_point = path_key[1]
                logging.debug(f"last point = {last_point}")
                last_time = path_key[2]
                logging.debug(f"last time = {last_time}")
                key_version = path_key[3]
                logging.debug(f"key version = {key_version}")
                optimal_cost, prev_last_point, prev_last_time, vehicle_order, prev_key_version = full_path_memo.get(path_key)
                logging.debug(f"optimal cost = {optimal_cost}")
                logging.debug(f"prev last point = {prev_last_point}")
                logging.debug(f"prev last time = {prev_last_time}")
                logging.debug(f"vehicle order = {vehicle_order}")
                logging.debug(f"prev key version = {prev_key_version}")
                optimal_path = [[0] for i in range(self.number_of_vehicles)]
                logging.debug(f"optimal path = {optimal_path}")
                current_time_vector = [[] for i in range(self.number_of_vehicles)]
            

                for i in range(self.number_of_vehicles):
                    current_time_vector[i] = max(path_key[2][i], self.start_times[last_point[i]]) + self.service_times[last_point[i]]+self.travel_times_array[last_point[i]][0]
                    logging.debug(f"current time vector[{i} = {max(path_key[2][i], self.start_times[last_point[i]])} +{self.service_times[last_point[i]]}+ {self.travel_times_array[last_point[i]][0]} = {current_time_vector[i]}")
                optimal_path_arrival_times = [[current_time_vector[i]] for i in vehicle_order]
                logging.debug(f"optimal path arrival times = {optimal_path_arrival_times}")
            

                while len(points_to_retrace) != 0:
                    last_point = path_key[1]
                    logging.debug(f"last point = {last_point}")
                    last_time = path_key[2]
                    logging.debug(f"last time = {last_time}")
                    key_version = path_key[3]
                    logging.debug(f"key version = {key_version}")
                    _, prev_last_point, prev_last_time, vehicle_order, prev_key_version = self.memo.get(path_key)
                    logging.debug(f"prev last point = {prev_last_point}")
                    logging.debug(f"prev last time = {prev_last_time}")
                    logging.debug(f"vehicle order = {vehicle_order}")
                    logging.debug(f"last point = {last_point}")
                    logging.debug(f"prev last point = {prev_last_point}")
                    logging.debug(f"prev key version = {prev_key_version}")
                
                    for i in range(self.number_of_vehicles):
                        if last_point[i] in prev_last_point:
                            continue
                        else:
                            point_to_remove = last_point[i] 
                            logging.debug(f"point to remove = {point_to_remove}")
                            optimal_path[vehicle_order[i]] = [point_to_remove] + optimal_path[vehicle_order[i]]
                            logging.debug(f"optimal path = {optimal_path}")
                            optimal_path_arrival_times[vehicle_order[i]] = [last_time[i]] + optimal_path_arrival_times[vehicle_order[i]]
                            logging.debug(f"optimal path arrival times = {optimal_path_arrival_times}")
                            res1 = [i for i in points_to_retrace if i not in [point_to_remove]]
                            points_to_retrace = tuple(sorted(res1))
                            logging.debug(f"points to retrace = {points_to_retrace}")
                            path_key = points_to_retrace, prev_last_point, prev_last_time, prev_key_version
                            logging.debug(f"path key = {path_key}")
                        
                    if len(points_to_retrace) == 1: #this means 0 is the only point left to be assigned to routes
                        logging.debug(f"points to retrace = {points_to_retrace}")
                        for i in range(self.number_of_vehicles):
                            optimal_path[i] = [0] + optimal_path[i]
                            optimal_path_arrival_times[i] = [0]+optimal_path_arrival_times[i]
                        logging.debug(f"optimal path = {optimal_path}")
                        logging.debug(f"optimal path arrival times = {optimal_path_arrival_times}")
                        point_to_remove = 0
                        res1 = [i for i in points_to_retrace if i not in [point_to_remove]]
                        points_to_retrace = tuple(sorted(res1))
                        logging.debug(f"points to retrace = {points_to_retrace}")
                     
            start1 = [[] for i in range(self.number_of_vehicles)]
            logging.debug(f"start1 = {start1}")
            end1 = [[] for i in range(self.number_of_vehicles)]
            logging.debug(f"end1 = {end1}")
            for j in range(self.number_of_vehicles):
                start1[j] = [self.start_times[i] for i in optimal_path[j]] 
                logging.debug(f"start1 = {start1}")
                end1[j] = [self.end_times[i] for i in optimal_path[j]]
                logging.debug(f"end1 = {end1}")
            logging.debug(f"optimal path arrival times = {optimal_path_arrival_times}")
            logging.debug(f"service times = {self.service_times}")
            logging.debug(f"start 1 = {start1}")
        
           
            optimal_path_departure_times = [[] for i in range(self.number_of_vehicles)]
            logging.debug(f"opt path depart = {optimal_path_departure_times}")
            for i in range(self.number_of_vehicles):
                logging.debug(f"for vehicle {i}:")
                for j in range(len(optimal_path[i])):
                    logging.debug(f"for job {optimal_path[i]}")
                    optimal_path_departure_times[i] = optimal_path_departure_times[i] + [max(optimal_path_arrival_times[i][j]+self.service_times[optimal_path[i][j]], start1[i][j])]
                    logging.debug(f"opt path depart = {optimal_path_departure_times}")
            logging.debug(f"optimal path departure times = {optimal_path_departure_times}")
            
            self.optimal_cost = optimal_cost
            self.optimal_path = optimal_path
            print("time check:")
            for i in range(self.number_of_vehicles):
                self.df1 = pd.DataFrame({'opt path[i]': optimal_path[i], 'start[i]': start1[i], 'arrival': optimal_path_arrival_times[i], 'departure[i]': optimal_path_departure_times[i], 'end[i]': end1[i] })
                print(f"{self.df1}")
        return self.optimal_path, self.optimal_cost

    def VRP_Solve(self, DUP, TW, SJA, DOM, T1, T2):
        
        self.all_points_set = [x for x in range(self.number_of_jobs)]
        self.special_values()
        self.queue.put((tuple([0]), tuple(self.prev_last_point), tuple(self.prev_time), self.key_version))
        self.memo [tuple([0]), tuple(self.prev_last_point), tuple(self.prev_time), self.key_version] = (tuple([self.prev_dist, self.prev_last_point, self.prev_time, self.vehicle_order, self.prev_key_version])) 
        counter = 0
        
        while self.queue.qsize() != 0:
            if (time.time() - self.t) > 3600:
                self.stopper = True
                break
                  
            self.prev_visited, self.prev_last_point, self.prev_time, self.prev_key_version = self.queue.get()
            
            while self.memo.get((tuple(self.prev_visited), tuple(self.prev_last_point), tuple(self.prev_time), self.prev_key_version)) == None:
                self.prev_visited, self.prev_last_point, self.prev_time, self.prev_key_version = self.queue.get()
            logging.debug(f"extending label {self.prev_visited}, {self.prev_last_point}, {self.prev_time}, {self.prev_key_version}")
            if len(self.prev_visited) != counter:
                print(f"progress check: {len(self.prev_visited)} cities of {self.number_of_jobs} cities visited, the running time is {time.time() - self.t}, and {len(self.memo)} labels have been created")
                counter = counter + 1
            self.prev_dist, _, _, self.vehicle_order, _ = self.memo[(tuple(self.prev_visited), tuple(self.prev_last_point), tuple(self.prev_time), self.prev_key_version)]
            logging.debug(f"previously visited set = {self.prev_visited}")
            logging.debug(f"previous last point = {self.prev_last_point}")
            logging.debug(f"previous time = {self.prev_time}")
            logging.debug(f"previous distance = {self.prev_dist}")
            logging.debug(f"vehicle order = {self.vehicle_order}")
            to_visit = [i for i in self.all_points_set if i not in self.prev_visited]
            
            logging.debug(f"to visit set = {to_visit}")
            for i in range(self.number_of_vehicles):
            
                logging.info(f"for vehicle {i}")
                self.new_last_point[i-1] = self.prev_last_point[i-1]
                self.new_dist[i-1] = self.prev_dist[i-1]
                self.new_time[i-1] = self.prev_time[i-1]
                for self.new_last_point[i] in to_visit:    
                    self.labels_considered = self.labels_considered + 1
                    logging.info(f"visiting job {self.new_last_point[i]}")
                    self.new_visited = tuple(sorted(list(self.prev_visited) + [self.new_last_point[i]]))
                    logging.debug(f"prev dist = {self.prev_dist[i]}")
                    logging.debug(f"new dist = {self.distances_array[self.prev_last_point[i]][self.new_last_point[i]]}")
                    self.new_dist[i] = float(np.round(self.prev_dist[i] + self.distances_array[self.prev_last_point[i]][self.new_last_point[i]],2))
                    self.new_time[i] = float(np.round(max(self.prev_time[i], self.start_times[self.prev_last_point[i]]) + self.service_times[self.prev_last_point[i]] + self.travel_times_array[self.prev_last_point[i]][self.new_last_point[i]],2))
                    logging.info(f"checking the new label ({self.new_visited},{self.new_last_point},{self.new_time}) with distances {self.new_dist}")
                    self.sorted_nlp, self.sorted_nt, self.sorted_nd, self.sorted_vo = zip(*sorted(zip(self.new_last_point, self.new_time, self.new_dist, self.vehicle_order)))
                    self.prefix_search2()
                    if DUP:
                        if not self.duplicate_label_check_VRP_update2():
                            continue
                    else:
                        logging.info(f"duplicate label check not being used")
                
               
                    if TW:
                        if not self.VRP_time_window_check():
                            continue
                    else:
                        logging.info(f"time window check not being used")

                    if SJA:
                        if not self.strong_jump_ahead_check():
                            continue
                
                    else:
                        logging.info(f"strong jump ahead test is not being used.")    
                    
                    logging.info(f"tests started")
                    if T2:
                        self.before_VRP()
                        if not self.VRP_test2():
                            logging.debug(f"tests ended, label rejected")
                            continue           
                    else:
                        logging.info(f"test 2 is not being used")
                
                    if T1:                    
                        if not self.VRP_test1A():
                            logging.debug(f"tests ended, label rejected")
                            continue

                    else:
                        logging.info(f"test 1 is not being used")
                
                    logging.info(f"tests ended")
                    
                    if DOM:
                        self.VRP_dominance_test_update4()
                    else:
                        if (tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version) in self.memo:
                            while (tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version) in self.memo:
                                self.key_version = self.key_version + 1
                        
                            self.memo[(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version)] = (self.sorted_nd, self.prev_last_point, self.prev_time, self.sorted_vo, self.prev_key_version)
                            self.queue.put((tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version))       
                            temp = self.prefix_memo[(tuple(self.new_visited), tuple(self.sorted_nlp))]
                            temp.append((tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version))
                            self.prefix_memo[(tuple(self.new_visited), tuple(self.sorted_nlp))] = temp
                        else:
                            self.memo[(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version)] = (self.sorted_nd, self.prev_last_point, self.prev_time, self.sorted_vo, self.prev_key_version)
                            self.queue.put((tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version))
                            if (tuple(self.new_visited), tuple(self.sorted_nlp)) in self.prefix_memo:
                                temp = self.prefix_memo[(tuple(self.new_visited), tuple(self.sorted_nlp))]
                                temp.append((tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version))
                                self.prefix_memo[(tuple(self.new_visited), tuple(self.sorted_nlp))] = temp
                            else:
                                self.prefix_memo[(tuple(self.new_visited), tuple(self.sorted_nlp))] = [(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt), self.key_version)]

        logging.debug(f"memo = {self.memo}")
        
        self.optimal_path, self.optimal_cost = self.retrace_optimal_path_VRP(self.memo, self.number_of_jobs)
    
        return self.optimal_path, self.optimal_cost, self.df1

#main solver
    def Solver(self, read_in_data, data, random_data, instances, timeframe, locationframe, servicetime, serviceframe, travel_times_multiplier, save_name, DUP, TW, T1, T2, T3, SJA, DOM):
        self.reset_problem()
        if read_in_data == True:
            self.read_in_data(data, travel_times_multiplier)
            print(self.df)
        else:
            logging.debug('no read in data given')
        if random_data == True:
            self.random_data_generator(instances, timeframe, locationframe, servicetime, serviceframe, travel_times_multiplier, save_name)
            print(self.df)
        else:
            logging.debug('no random data generated')
        if self.number_of_vehicles == 1:
            logging.info(f"TSP situation")
            self.t = time.time()
            self.Dumas_TSPTW_Solve(T1, T2, T3)
            self.run_time = round(time.time() - self.t, 3)
        else:
            logging.debug(f"VRP situation")
            self.t = time.time()
            self.VRP_Solve(DUP, TW, SJA, DOM, T1, T2)
            self.run_time = round(time.time() - self.t, 3)
        if self.stopper == False:            
            print(f"the memo length is {len(self.memo)}")
            if self.number_of_vehicles == 1:
                print(f"the length of the test 3 rejected labels is {self.test3_rejected_labels}")
            print(f"Found optimal path in {self.run_time} seconds.") 
            print(f"Optimal cost: {self.optimal_cost}, optimal path: {self.optimal_path}")
            print(f"duplicate label check rejected {self.dup_lab_rejected} labels.")
            print(f"time window check rejected {self.TW_rejected} labels.")
            print(f"jump ahead check rejected {self.jump_ahead_rejected} labels.")
            print(f"test 2 rejected {self.test2_rejected} labels.")
            print(f"test 1 rejected {self.test1_rejected} labels.")
            print(f"dominance check rejected {self.dom_lab_rejected} labels.")
            print(f"{self.labels_considered} were considered")
            
        return

    
#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
a = VRP_Problem(number_of_vehicles = 2)

names = ['VRP_testing_05_jobs_1','VRP_testing_05_jobs_2', 'VRP_testing_05_jobs_3','VRP_testing_05_jobs_4', 'VRP_testing_05_jobs_5','VRP_testing_10_jobs_1', 'VRP_testing_10_jobs_2', 'VRP_testing_10_jobs_3', 'VRP_testing_10_jobs_4', 'VRP_testing_10_jobs_5','VRP_testing_15_jobs_1', 'VRP_testing_15_jobs_2', 'VRP_testing_15_jobs_3', 'VRP_testing_15_jobs_4', 'VRP_testing_15_jobs_5','VRP_testing_20_jobs_1', 'VRP_testing_20_jobs_2', 'VRP_testing_20_jobs_3', 'VRP_testing_20_jobs_4', 'VRP_testing_20_jobs_5','VRP_testing_25_jobs_1', 'VRP_testing_25_jobs_2', 'VRP_testing_25_jobs_3', 'VRP_testing_25_jobs_4', 'VRP_testing_25_jobs_5']
names05 = ['VRP_testing_05_jobs_1','VRP_testing_05_jobs_2', 'VRP_testing_05_jobs_3','VRP_testing_05_jobs_4', 'VRP_testing_05_jobs_5']
names10 = ['VRP_testing_10_jobs_1', 'VRP_testing_10_jobs_2', 'VRP_testing_10_jobs_3', 'VRP_testing_10_jobs_4', 'VRP_testing_10_jobs_5']
names15 = ['VRP_testing_15_jobs_1', 'VRP_testing_15_jobs_2', 'VRP_testing_15_jobs_3', 'VRP_testing_15_jobs_4', 'VRP_testing_15_jobs_5']
names20 = ['VRP_testing_20_jobs_1', 'VRP_testing_20_jobs_2', 'VRP_testing_20_jobs_3', 'VRP_testing_20_jobs_4', 'VRP_testing_20_jobs_5']
names25 = ['VRP_testing_25_jobs_1', 'VRP_testing_25_jobs_2', 'VRP_testing_25_jobs_3', 'VRP_testing_25_jobs_4', 'VRP_testing_25_jobs_5']
big_names = ['VRP_testing_15_jobs_1', 'VRP_testing_15_jobs_2', 'VRP_testing_15_jobs_3', 'VRP_testing_15_jobs_4', 'VRP_testing_15_jobs_5', 'VRP_testing_20_jobs_1', 'VRP_testing_20_jobs_2', 'VRP_testing_20_jobs_3', 'VRP_testing_20_jobs_4', 'VRP_testing_20_jobs_5', 'VRP_testing_25_jobs_1', 'VRP_testing_25_jobs_2', 'VRP_testing_25_jobs_3', 'VRP_testing_25_jobs_4', 'VRP_testing_25_jobs_5']


t1 = [False, False, True, True]
t2 = [False, True, False, True]
for i in range(len(names15)):
    for j in {0,1,2,3}:
        print(f"data set = {names15[i]}")
        print(f"TW = True")
        print(f"DUP = True")
        print(f"SJA = True")
        print(f"DOM = True")
        print(f"T1 = {t1[j]}")
        print(f"T2 = {t2[j]}")
        pr = cProfile.Profile()
        pr.enable()
        a.Solver(read_in_data = True, data = names15[i], random_data = False, instances = 7, timeframe = 2000, locationframe = 100, servicetime = True, serviceframe = 25, travel_times_multiplier = 1, save_name = big_names[i], DUP = True, TW = True, T1 = t1[j], T2 = t2[j], T3 = False, SJA = True, DOM = True)
        print(f"###COMPLETE_RESULTS:, {names10[i]}, T1 = {t1[j]}, T2 = {t2[j]}, {a.run_time}, {a.optimal_cost}, {a.optimal_path}, {len(a.memo)}")
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(.1)
        print(s.getvalue())
        
        
t1 = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
t2 = [False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, True]
dom = [False, False, False, False, True, True, True, True, False, False, False, False, True, True, True, True, False, False, False, False, True, True, True, True, False, False, False, False, True, True, True, True]
dup = [False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, True]
sja = [False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True, False, True]

input()

for i in range(len(names)):
    for j in range(len(dom)):    
        print(f"data set = {names[i]}")
        print(f"TW = True")
        print(f"DUP = {dup[j]}")
        print(f"SJA = {sja[j]}")
        print(f"T1 = {t1[j]}")
        print(f"T2 = {t2[j]}")
        print(f"DOM = {dom[j]}")
        if dup[j] == False and sja[j] == False and t1[j] == False and t2[j] == False and dom[j] == False:
            print("this instance is skipped")
        elif dup[j] == False and sja[j] == False and t1[j] == False and t2[j] == True and dom[j] == False:
            print("this instance is skipped")
        elif dup[j] == False and sja[j] == False and t1[j] == True and t2[j] == False and dom[j] == False:
            print("this instance is skipped")
        elif dup[j] == False and sja[j] == False and t1[j] == True and t2[j] == True and dom[j] == False:
            print("this instance is skipped")
        else:
            a.Solver(read_in_data = True, data = names[i], random_data = False, instances = 7, timeframe = 2000, locationframe = 100, servicetime = True, serviceframe = 25, travel_times_multiplier = 1, save_name = names[i], DUP = dup[j], TW = True, T1 = t1[j], T2 = t2[j], T3 = False, SJA = sja[j], DOM = dom[j])
            print(f"###COMPLETE_RESULTS:, {names[i]}, {dup[j]}, {sja[j]}, {t1[j]}, {t2[j]}, {dom[j]}, {a.run_time}, {a.optimal_cost}, {a.optimal_path}, {len(a.memo)}")
