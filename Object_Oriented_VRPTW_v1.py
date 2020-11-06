import csv
import time
import numpy as np
np.random.seed(42)
import random
import pandas as pd
import logging
import copy

class VRP_Problem:
    def __init__(self, number_of_vehicles):
        self.number_of_vehicles = number_of_vehicles
        self.number_of_jobs = None
        self.job_locations = []
        self.start_times = []
        self.end_times = []
        self.service_times = []
        self.memo = {}
        self.queue = []
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
        #self.first = None
        self.first = [0 for i in range(self.number_of_vehicles)]
        self.all_points_set = {}
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
        self.sorted_nlp = None
        self.sorted_nt = [0 for i in range(self.number_of_vehicles)]
        self.sorted_nd = [0 for i in range(self.number_of_vehicles)]
        self.sorted_vo = [i for i in range(self.number_of_vehicles)]

    def reset_problem(self):
        self.number_of_jobs = None
        self.number_of_jobs = None
        self.job_locations = []
        self.start_times = []
        self.end_times = []
        self.service_times = []
        self.memo = {}
        self.queue = []
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
        #self.first = None
        self.first = [0 for i in range(self.number_of_vehicles)]
        self.all_points_set = {}
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
        self.sorted_nlp = None
        self.sorted_nt = [0 for i in range(self.number_of_vehicles)]
        self.sorted_nd = [0 for i in range(self.number_of_vehicles)]
        self.sorted_vo = [i for i in range(self.number_of_vehicles)]
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
        self.all_points_set = set(range(self.number_of_jobs))
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
            self.all_points_set = set(range(self.number_of_jobs))
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
            self.all_points_set = set(range(self.number_of_jobs))
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

    def time_window_check(self):
        logging.info('time window check')
        if self.new_time > self.end_times[self.new_last_point]:
            TW = False
            logging.info(f"TW infeasible, {self.new_time} > {self.end_times[self.new_last_point]}")
        else: 
            TW = True
            logging.info(f"TW feasible, {self.new_time} <= {self.end_times[self.new_last_point]}")
        return TW

    def VRP_time_window_check(self):
        logging.info(f"start TW check")  
        for i in range(self.number_of_vehicles):
            logging.debug(f"checking vehcile {i}")
            logging.debug(f"NT = {self.sorted_nt[i]}")
            logging.debug(f"ET = {self.end_times[self.sorted_nlp[i]]}")
            if self.sorted_nt[i] <= self.end_times[self.sorted_nlp[i]]:
                self.TW[i] = True
                logging.debug("TW[{i}] = true")                
            else:
                self.TW[i] = False
                logging.debug("TW[{i}] = false")                
        logging.debug(f"TW = {self.TW}")
        if False in self.TW:
            TW_test = False
        else:
            TW_test = True
        logging.info(f"time window test = {TW_test}")
        if TW_test == False:
            self.TW_rejected = self.TW_rejected + 1
        return TW_test

    def dumas_before(self, x):
        for i in self.all_points_set:
                if self.start_times[i]+self.service_times[i]+self.travel_times_array[i][x] > self.end_times[x]:
                    self.dumas_before_sets[i].append(x)
                    logging.debug(self.dumas_before_sets)
                else:
                    continue
        return self.dumas_before_sets
    
    def before_VRP(self):
        Y = [self.dumas_before_sets[x] for x in self.sorted_nlp]
        self.VRP_before_set = set(Y[0]).intersection(*Y)
        return self.VRP_before_set

    def dumas_test2(self):
        logging.debug(f"before({self.new_last_point})={self.before_sets[self.new_last_point]}")
        logging.debug(f"S = {self.new_visited}")
        if set(self.before_sets[self.new_last_point]).issubset(set(self.new_visited)) == False:
            test2 = False
            logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) fails test 2")
        else: 
            test2 = True
            logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) passes test 2")
        return test2

    def VRP_test2(self):
        self.before_VRP()
        logging.debug(f"before({self.sorted_nlp})={self.VRP_before_set}")
        logging.debug(f"S = {self.new_visited}")
        if self.VRP_before_set.issubset(set(self.new_visited)) == False:
            test2 = False
            logging.info(f"({self.new_visited}, {self.sorted_nlp}, {self.sorted_nt}) fails test 2")
        else: 
            test2 = True
            logging.info(f"({self.new_visited}, {self.sorted_nlp}, {self.sorted_nt}) passes test 2")
        if test2 == False:
            self.test2_rejected = self.test2_rejected + 1
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

    #the first function determines the earliest time a vehicle arrives at the jobs in V_i while servicing the jobs in S_i
    def VRP_first(self):
        sorted_nlp1, sorted_nt1 = zip(*sorted(zip(self.new_last_point, self.new_time)))
        #res1 = {k: v for k, v in self.memo.items() if k[0]==self.new_visited and k[1]==sorted_nlp1}
        #logging.debug(f"key search ({self.new_visited}, {self.new_last_point}, _) = {res1}")
        logging.debug(f"key search ({self.new_visited}, {self.new_last_point}, _) = {self.label_check}")
        res2 = {value: key for key, value in self.label_check.items()}
        if len(self.label_check) !=0: #len(res1) != 0:
            res2 = [key for key in self.label_check.keys()]
            logging.debug(f"relevent keys = {res2}")
            res3 = [res2[i][2] for i in range(len(res2))]      
            logging.debug(f"relevent times = {res3}")
        else:
            res3=[]
        if len(res3) == 0:
            logging.debug(f"current times are earliest times = {sorted_nt1}")
            for i in range(self.number_of_vehicles):
                self.first[i] = sorted_nt1[i]
        else:
            #res4 = []
            for j in range(len(res3)):
                logging.debug(f"key {j} = {res3[j]}")
                for i in range(self.number_of_vehicles):
                    #res4 = res4 + [min(res3[j][i])]
                    #logging.debug(f"minimum values = {res3}")
                    self.first[i] = min(res3[j][i], sorted_nt1[i])
                    logging.debug(f"first = {self.first}")
        return

    def dumas_test1(self):
        
        if len(self.all_points_set.difference(self.new_visited)) == 0:
            test1 = True
        elif self.first > min(self.LDT_array[j][self.new_last_point] for j in self.all_points_set.difference(self.new_visited)): 
            logging.debug(f"first = {self.first} > {min(self.LDT_array[j][self.new_last_point] for j in self.all_points_set.difference(self.new_visited))}")
            logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) fails test 1")
            test1 = False
        else:
            logging.debug(f"first = {self.first} > {min(self.LDT_array[j][self.new_last_point] for j in self.all_points_set.difference(self.new_visited))}")
            logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) passes test 1")
            test1 = True
        return test1

    def VRP_test1A(self):
        unreachable_points = self.all_points_set.difference(self.new_visited)
        logging.debug(f"unreachable points = {unreachable_points}")
        if len(unreachable_points) != 0:
            self.VRP_first()
            #_, first1 = zip(*sorted(zip(self.vehicle_order, self.first)))
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
                        unreachable_points = unreachable_points.difference({j})
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
        return test1

    def VRP_test1(self):
        
        if len(self.all_points_set.difference(self.new_visited)) == 0:
            test1 = True
            logging.debug(f"test1 = {test1}")
        else:            
            self.VRP_first()
            _, first1 = zip(*sorted(zip(self.vehicle_order, self.first)))
            logging.debug(f"first1 = {first1}")
            #input()
            T1_LDT = [0 for i in range(self.number_of_vehicles)]
            logging.debug(f"T1_LDT = {T1_LDT}")
            #input()
            for i in range(self.number_of_vehicles):
                T1_LDT[i] = int(min(self.LDT_array[j][self.new_last_point[i]] for j in self.all_points_set.difference(self.new_visited)))
            T1_LDT=tuple(T1_LDT)
            logging.debug(f"T1_LDT = {T1_LDT}")
            #input()
            T1_outcome = [(first1 <= T1_LDT) for first1, T1_LDT in zip(first1, T1_LDT)]
            logging.debug(f"T1_outcome = {T1_outcome}")
            #input()
            check = all(T1_outcome)
            logging.debug(f"check = {check}")        
            if not check:
                #logging.debug(f"first = {self.first} > {min(self.LDT_array[j][self.new_last_point] for j in self.all_points_set.difference(self.new_visited))}")
                logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) fails test 1")
                test1 = False
                logging.debug(f"test1 = {test1}")
            else:
                #logging.debug(f"first = {self.first} <= {min(self.LDT_array[j][self.new_last_point] for j in self.all_points_set.difference(self.new_visited))}")
                logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) passes test 1")
                test1 = True
                logging.debug(f"test1 = {test1}")
        return test1

    def dumas_test3(self):
        logging.debug(f"adding to restricted labels")
        for x in self.all_points_set.difference(self.new_visited):
                new_visited1 = sorted(self.new_visited)
                new_last_point1 = self.new_last_point
                new_time1 = self.new_time
                if self.new_time <= self.LDT_array[x][self.new_last_point]:
                    new_visited1 = [x] + list(new_visited1)
                    new_visited1 = tuple(sorted(new_visited1))
                    new_last_point1 = x
                    new_time1 = self.new_time+self.service_times[self.new_last_point]+self.travel_times_array[self.new_last_point][x]
                    for y in self.all_points_set.difference(new_visited1):
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
            points_to_retrace = tuple(sorted(set(points_to_retrace).difference({last_point})))

            while next_to_last_point is not None:
                last_point = next_to_last_point
                end_time = prev_time
                path_key = (points_to_retrace, last_point, prev_time)
                _, next_to_last_point, prev_time = memo[path_key]
                optimal_path = [last_point] + optimal_path
                optimal_path_arrival_times = [end_time] + optimal_path_arrival_times
                points_to_retrace = tuple(sorted(set(points_to_retrace).difference({last_point})))

            start1 = [self.start_times[i] for i in optimal_path]
            end1 = [self.end_times[i] for i in optimal_path]
            optimal_path_departure_times = [max(optimal_path_arrival_times[i]+self.service_times[i], start1[i]) for i in range(len(self.distances_array))]
            self.optimal_path = optimal_path
            self.optimal_cost = optimal_cost
            self.df1 = pd.DataFrame({'opt path': optimal_path, 'start': start1, 'arrival': optimal_path_arrival_times, 'departure': optimal_path_departure_times, 'end': end1 })
            print("time check:")
            print(self.df1)
        return optimal_path, optimal_cost

    def retrace_optimal_path_VRP(self, memo: dict, n: int) -> [[int], float]:
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
                full_path_memo[x] = (tuple(current_cost_vector), full_path_memo.get(x)[1], full_path_memo.get(x)[2], full_path_memo.get(x)[3])
                logging.debug(f"full path memo[{x}] = {full_path_memo[x]}")
        logging.debug(f"updated cost full path memo = {full_path_memo}")     
        print(f"full path memo = {full_path_memo}")
        

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
            optimal_cost, prev_last_point, prev_last_time, vehicle_order = full_path_memo.get(path_key)
            logging.debug(f"optimal cost = {optimal_cost}")
            logging.debug(f"prev last point = {prev_last_point}")
            logging.debug(f"prev last time = {prev_last_time}")
            logging.debug(f"vehicle order = {vehicle_order}")
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
                _, prev_last_point, prev_last_time, vehicle_order = self.memo.get(path_key)
                logging.debug(f"prev last point = {prev_last_point}")
                logging.debug(f"prev last time = {prev_last_time}")
                logging.debug(f"vehicle order = {vehicle_order}")
                logging.debug(f"last point = {last_point}")
                logging.debug(f"prev last point = {prev_last_point}")
                #input()
                for i in range(self.number_of_vehicles):
                    if last_point[i] in prev_last_point:
                        continue
                    else:
                        point_to_remove = last_point[i] #LEFTOFF HERE NOT ADDING POINTS TO THE CORRECT ROUTE
                        logging.debug(f"point to remove = {point_to_remove}")
                        optimal_path[vehicle_order[i]] = [point_to_remove] + optimal_path[vehicle_order[i]]
                        logging.debug(f"optimal path = {optimal_path}")
                        optimal_path_arrival_times[vehicle_order[i]] = [last_time[i]] + optimal_path_arrival_times[vehicle_order[i]]
                        logging.debug(f"optimal path arrival times = {optimal_path_arrival_times}")
                        points_to_retrace = tuple(sorted(set(points_to_retrace).difference({point_to_remove})))
                        logging.debug(f"points to retrace = {points_to_retrace}")
                        path_key = points_to_retrace, prev_last_point, prev_last_time
                        logging.debug(f"path key = {path_key}")
                        #input()
                if len(points_to_retrace) == 1: #this means 0 is the only point left to be assigned to routes
                    logging.debug(f"points to retrace = {points_to_retrace}")
                    for i in range(self.number_of_vehicles):
                        optimal_path[i] = [0] + optimal_path[i]
                        optimal_path_arrival_times[i] = [0]+optimal_path_arrival_times[i]
                    logging.debug(f"optimal path = {optimal_path}")
                    logging.debug(f"optimal path arrival times = {optimal_path_arrival_times}")
                    point_to_remove = 0
                    points_to_retrace = tuple(sorted(set(points_to_retrace).difference({point_to_remove})))
                    logging.debug(f"points to retrace = {points_to_retrace}")
                    #input()    
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
        #input()
           
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

    def dominance_test(self):
        logging.info(f"starting dominance check:")
        if len({k: v for k, v in self.memo.items() if k[0]==self.new_visited and k[1]==self.new_last_point}) != 0: #possible dominated situation            
            if (self.new_time <= self.first) and (self.new_dist <= self.memo[(self.new_visited, self.new_last_point, self.first)][0]): #time and cost improvement, replace label
                del self.memo[(self.new_visited, self.new_last_point, self.first)]
                self.memo[(self.new_visited, self.new_last_point, self.new_time)] = (self.new_dist, self.prev_last_point, self.prev_time)
                self.queue.remove((self.new_visited, self.new_last_point, self.first))
                self.queue += [(self.new_visited, self.new_last_point, self.new_time)]
                logging.info(f"label ({self.new_visited}, {self.new_last_point}, {self.new_time}) dominated, case 1, replaces label ({self.new_visited},{self.new_last_point},{self.first})")
            elif (self.new_time == self.first) and (self.new_dist < self.memo[(self.new_visited, self.new_last_point, self.first)][0]): #same time, cost improvement, replace old label with new
                self.memo[(self.new_visited, self.new_last_point, self.first)] = (self.new_dist, self.prev_last_point, self.prev_time)
                self.queue += [(self.new_visited, self.new_last_point, self.new_time)]
                logging.info(f"label ({self.new_visited}, {self.new_last_point}, {self.new_time}) dominated, case 2, same time, better distance, updates old label with new distance")
            elif (self.new_time < self.first) and (self.new_dist >= self.memo[(self.new_visited, self.new_last_point, self.first)][0]): #time improvement only, add new label
                self.memo[(self.new_visited, self.new_last_point, self.new_time)] = (self.new_dist, self.prev_last_point, self.prev_time)
                self.queue += [(self.new_visited, self.new_last_point, self.new_time)]
                logging.info(f"label ({self.new_visited}, {self.new_last_point}, {self.new_time}) dominated, case 3, better time, worse cost, adds label")
            elif (self.new_time >= self.first) and (self.new_dist  < self.memo[(self.new_visited, self.new_last_point, self.first)][0]): #cost improvement only, add new label
                self.memo[(self.new_visited, self.new_last_point, self.new_time)] = (self.new_dist, self.prev_last_point, self.prev_time)
                self.queue += [(self.new_visited, self.new_last_point, self.new_time)]
                logging.info(f"label ({self.new_visited}, {self.new_last_point}, {self.new_time}) dominated, case 4, slower time, better cost, adds label")
            else:
                logging.info(f"label ({self.new_visited}, {self.new_last_point}, {self.new_time}) dominated, case 5, no label created")
                #if (new_time >= first) and (new_dist >= memo[(new_visited, new_last_point, first)][0]): #no improvement, no label added
                #queue += [(new_visited, new_last_point, new_time)]                 
        else: #len({k: v for k, v in memo.items() if k[0]==new_visited and k[1]==new_last_point}) == 0: #new label
            self.memo[(self.new_visited, self.new_last_point, self.new_time)] = (self.new_dist, self.prev_last_point, self.prev_time)
            self.queue += [(self.new_visited, self.new_last_point, self.new_time)]
            logging.info(f"no (S,i) label exists, {self.new_visited, self.new_last_point, self.new_time} added") 
        logging.debug(f"the queue is {self.queue}")
        logging.debug(f"the memo is {self.memo}")
        return

    def VRP_dominance_test(self):
        logging.info(f"starting dominance check:")
        sorted_nlp, sorted_nt, sorted_nd, sorted_vo = zip(*sorted(zip(self.new_last_point, self.new_time, self.new_dist, self.vehicle_order)))
        self.VRP_first()
        sorted_first = self.first
        if len({k: v for k, v in self.memo.items() if k[0]==self.new_visited and k[1]==sorted_nlp}) != 0: #possible dominated situation            
            prev_lab_dist = self.memo[(self.new_visited, tuple(sorted_nlp), tuple(sorted_nt))][0]
            logging.debug(f"current label's distances = {prev_lab_dist}")
            outcome1 = [(sorted_nt <= sorted_first) for sorted_nt, sorted_first in zip(sorted_nt, sorted_first)]
            outcome2 = [(sorted_nd <= prev_lab_dist) for sorted_nd, prev_lab_dist in zip(sorted_nd, prev_lab_dist)]
            outcome3 = [(sorted_nt < sorted_first) for sorted_nt, sorted_first in zip(sorted_nt, sorted_first)]
            outcome4 = [(sorted_nd >= prev_lab_dist) for sorted_nd, prev_lab_dist in zip(sorted_nd, prev_lab_dist)]
            outcome5 = [(sorted_nt >= sorted_first) for sorted_nt, sorted_first in zip(sorted_nt, sorted_first)]
            outcome6 = [(sorted_nd < prev_lab_dist) for sorted_nd, prev_lab_dist in zip(sorted_nd, prev_lab_dist)]
            if all(outcome1) and all(outcome2): #time and cost improvement, replace label
                del self.memo[(tuple(self.new_visited), tuple(sorted_nlp), tuple(sorted_first))]
                self.memo[(tuple(self.new_visited), tuple(sorted_nlp), tuple(sorted_nt))] = (sorted_nd, self.prev_last_point, self.prev_time, sorted_vo)
                self.queue.remove((tuple(self.new_visited), tuple(sorted_nlp), tuple(sorted_first)))
                self.queue.append((tuple(self.new_visited), tuple(sorted_nlp), tuple(sorted_nt)))
                logging.info(f"label ({self.new_visited}, {sorted_nlp}, {sorted_nt}) dominated, case 1, replaces label ({self.new_visited},{self.new_last_point},{sorted_first})")
                #input()
            elif all(outcome3) and all(outcome4): #time improvement only, add new label
                self.memo[(tuple(self.new_visited), tuple(sorted_nlp), tuple(sorted_nt))] = (sorted_nd, self.prev_last_point, self.prev_time, sorted_vo)
                self.queue.append((tuple(self.new_visited), tuple(sorted_nlp), tuple(sorted_nt)))
                logging.info(f"label ({self.new_visited}, {sorted_nlp}, {sorted_nt}) dominated, case 3, better time, worse cost, adds label")
                #input()
                                
            elif all(outcome5) and all(outcome6): #cost improvement only, add new label
                self.memo[(tuple(self.new_visited), tuple(sorted_nlp), tuple(sorted_nt))] = (sorted_nd, self.prev_last_point, self.prev_time, sorted_vo)
                self.queue.append((self.new_visited, sorted_nlp, sorted_nt))
                logging.info(f"label ({self.new_visited}, {sorted_nlp}, {sorted_nt}) dominated, case 4, slower time, better cost, adds label")
                #input()
            else:
                logging.info(f"label ({self.new_visited}, {sorted_nlp}, {sorted_nt}) dominated, case 5, no label created")
                #input()                 
        else: 
            self.memo[(tuple(self.new_visited), tuple(sorted_nlp), tuple(sorted_nt))] = (sorted_nd, self.prev_last_point, self.prev_time, sorted_vo)
            self.queue.append((tuple(self.new_visited), tuple(sorted_nlp), tuple(sorted_nt)))
            logging.info(f"no (S,V_i) label exists, {self.new_visited, sorted_nlp, sorted_nt} added") 
            #input()
        return

    #dominance test checks to see if the current label has is totally dominated by any label already stored.  If so the current label is not added.  Otherwise the current label is added and then it checks to see if the current label totally dominates any existing label.  If so those existing labels that are totally dominated are deleted.  If no existing labels are dominated nothing happens.
    def VRP_dominance_test_update(self):
        logging.info(f"starting dominance check:")
        #sorted_nlp, sorted_nt, sorted_nd, sorted_vo = zip(*sorted(zip(self.new_last_point, self.new_time, self.new_dist, self.vehicle_order)))
        #dom_lab = {k: v for k, v in self.memo.items() if k[0]==self.new_visited and k[1]==sorted_nlp}
        dom_lab = self.label_check
        logging.debug(f"possible dominated labels = {dom_lab}")
        if len(dom_lab) != 0: #possible dominated situation            
            dom_lab_keys = [key for key in dom_lab.keys()]
            dom_lab_times = [dom_lab_keys[i][2] for i in range(len(dom_lab_keys))]
            logging.debug(f"possible dominated label times = {dom_lab_times}")
            dom_lab_values = [value for value in dom_lab.values()]
            dom_lab_dist = [dom_lab_values[i][0] for i in range(len(dom_lab_values))]
            logging.debug(f"possible dominated label distances = {dom_lab_dist}")
            time_check = [[False for j in range(self.number_of_vehicles)] for i in range(len(dom_lab))]
            dist_check = [[False for j in range(self.number_of_vehicles)] for i in range(len(dom_lab))]
            for j in range(self.number_of_vehicles):
                for i in range(len(dom_lab)):
                    if self.sorted_nt[j] <= dom_lab_times[i][j]:
                        time_check[i][j] = True
                        logging.debug(f"time check = {time_check}")
                    else:
                        continue
                    if self.sorted_nd[j] <= dom_lab_dist[i][j]:
                        dist_check[i][j] = True
                        logging.debug(f"dist check = {dist_check}")
                    else:
                        continue
            label_dom_by_time = [not any(time_check[i]) for i in range(len(dom_lab))]
            label_dom_by_dist = [not any(dist_check[i]) for i in range(len(dom_lab))]
            label_to_delete = []
            if all(label_dom_by_time) and all(label_dom_by_dist):
                logging.info(f"the new label ({self.new_visited}, {self.sorted_nlp}, {self.sorted_nt}) is totally dominated by all existing labels, so it is not added")
            else:
                for i in range(len(dom_lab)):
                    if all(time_check[i]) and all(dist_check[i]):
                        label_to_delete = label_to_delete + [i]
                if len(label_to_delete) != 0:
                    #logging.info(f"the new label ({self.new_visited}, {sorted_nlp}, {sorted_nt}) totally dominates the existing labels {dom_lab[i] for i in label_to_delete}") #this does not display the correct message?
                    for i in label_to_delete:
                        del self.memo[(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(dom_lab_times[i]))]
                        self.queue.remove((tuple(self.new_visited), tuple(self.sorted_nlp), tuple(dom_lab_times[i])))
                        self.deleted_labels.append((tuple(self.new_visited), tuple(self.sorted_nlp), tuple(dom_lab_times[i])))
                        logging.info(f"the existing label {self.new_visited}, {self.sorted_nlp}, {dom_lab_times[i]} is totally dominated by the current label so it is deleted.")
                        self.dom_lab_rejected = self.dom_lab_rejected + 1
                    logging.info(f"the new label ({self.new_visited}, {self.sorted_nlp}, {self.sorted_nt}) is NOT totally dominated by existing labels, so it is added")
                    self.memo[(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt))] = (self.sorted_nd, self.prev_last_point, self.prev_time, self.sorted_vo)
                    self.queue.append((tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt)))
                    self.added_labels.append((tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt)))
        else: 
            self.memo[(tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt))] = (self.sorted_nd, self.prev_last_point, self.prev_time, self.sorted_vo)
            self.queue.append((tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt)))
            self.added_labels.append((tuple(self.new_visited), tuple(self.sorted_nlp), tuple(self.sorted_nt)))
            logging.info(f"no (S,V_i) label exists, {self.new_visited, self.sorted_nlp, self.sorted_nt} added") 
               
        return
    
    #this checks to see if the exact same label is already in the memo, if so, the current label is abandoned otherwise the other checks proceed
    def duplicate_label_check_VRP(self):
        #sorted_nlp, sorted_nt, sorted_nd, sorted_vo = zip(*sorted(zip(self.new_last_point, self.new_time, self.new_dist, self.vehicle_order)))
        dup_label_check = {k: v for k, v in self.label_check.items() if k[0]==self.new_visited and k[1]==self.sorted_nlp and k[2]==self.sorted_nt}
        logging.info(f"starting duplicate label check for label ({dup_label_check}, {self.sorted_nlp}, {self.sorted_nt}) with values ({self.sorted_nd}, {self.prev_last_point}, {self.prev_time}, _):")
        
        if len(dup_label_check) == 0:
            dup_lab_check = True
            logging.info(f"the label ({self.new_visited}, {self.sorted_nlp}, {self.sorted_nt}) IS NOT a duplicate label")
        else:
            #test_sorted_nd, test_prev_last_point, test_prev_time, _ = self.memo.get((self.new_visited, sorted_nlp, sorted_nt)) #do I need all of these for a duplicate label ... would just the same distance be enough?
            test_sorted_nd, _, _, _ = dup_label_check.get((self.new_visited, self.sorted_nlp, self.sorted_nt))
            logging.debug(f"test sorted new distance = {test_sorted_nd} and sorted new distance = {self.sorted_nd}")
            #logging.debug(f"test prev last point = {test_prev_last_point} and prev last point = {self.prev_last_point}")
            #logging.debug(f"test prev time = {test_prev_time} and prev time = {self.prev_time}")
            if test_sorted_nd == self.sorted_nd: #and test_prev_last_point == self.prev_last_point and test_prev_time == self.prev_time:
                dup_lab_check = False
                logging.info(f"the label ({self.new_visited}, {self.sorted_nlp}, {self.sorted_nt}) IS a duplicate label")
            else:
                dup_lab_check = True
                logging.info(f"the label ({self.new_visited}, {self.sorted_nlp}, {self.sorted_nt}) IS NOT a duplicate label")
        if dup_lab_check == False:
            self.dup_lab_rejected = self.dup_lab_rejected + 1
        return dup_lab_check

    def Dumas_TSPTW_Solve(self, T1, T2, T3):
        self.dumas_before_sets = [[ ] for y in self.all_points_set]
        self.special_values()
        
        self.memo = {(tuple([i]), i, 0): tuple([0, None, 0]) for i in range(self.number_of_jobs)} 
        self.queue = [(tuple([i]), i, 0) for i in range(self.number_of_jobs)]
        while self.queue: 
            self.prev_visited, self.prev_last_point, self.prev_time = self.queue.pop(0)
            self.prev_dist, _, _ = self.memo[(self.prev_visited, self.prev_last_point, self.prev_time)]
            logging.info(f"extending from ({self.prev_visited}, {self.prev_last_point}, {self.prev_time})")
            to_visit = self.all_points_set.difference(set(self.prev_visited))
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
                

        #end while
        self.optimal_path, self.optimal_cost = self.retrace_optimal_path_TSPTW(self.memo, self.number_of_jobs)
    
        return self.optimal_path, self.optimal_cost, self.df1
    
    def VRP_Solve(self, T1, T2):
        self.all_points_set = {x for x in range(self.number_of_jobs)}
        self.special_values()
        #input()
        self.queue = [(tuple([0]), tuple(self.prev_last_point), tuple(self.prev_time))]
        self.memo [tuple([0]), tuple(self.prev_last_point), tuple(self.prev_time)] = (tuple([self.prev_dist, self.prev_last_point, self.prev_time, self.vehicle_order])) 
        while self.queue:
            self.prev_visited, self.prev_last_point, self.prev_time = self.queue.pop(0)
            logging.info(f"the queue is now {len(self.queue)} long")
            logging.info(f"the memo is now {len(self.memo)} long")
            logging.debug(f"{len(self.added_labels)} have been added")
            logging.debug(f"{len(self.deleted_labels)} have been deleted")
            logging.debug(f"extending label {self.prev_visited}, {self.prev_last_point}, {self.prev_time}")
            self.prev_dist, _, _, self.vehicle_order = self.memo[(tuple(self.prev_visited), tuple(self.prev_last_point), tuple(self.prev_time))]
            logging.debug(f"previously visited set = {self.prev_visited}")
            logging.debug(f"previous last point = {self.prev_last_point}")
            logging.debug(f"previous time = {self.prev_time}")
            logging.debug(f"previous distance = {self.prev_dist}")
            logging.debug(f"vehicle order = {self.vehicle_order}")
            to_visit = self.all_points_set.difference(set(self.prev_visited))
            logging.debug(f"to visit set = {to_visit}")
            for i in range(self.number_of_vehicles):
                logging.info(f"for vehicle {i}")
                self.new_last_point[i-1] = self.prev_last_point[i-1] #Is this the problem?
                self.new_dist[i-1] = self.prev_dist[i-1] #Is this the problem?
                self.new_time[i-1] = self.prev_time[i-1] #Is this the problem?
                for self.new_last_point[i] in to_visit:    
                    logging.info(f"visiting job {self.new_last_point[i]}")
                    self.new_visited = tuple(sorted(list(self.prev_visited) + [self.new_last_point[i]]))
                    self.new_dist[i] = self.prev_dist[i] + self.distances_array[self.prev_last_point[i]][self.new_last_point[i]]
                    self.new_time[i] = max(self.prev_time[i], self.start_times[self.prev_last_point[i]]) + self.service_times[self.prev_last_point[i]] + self.travel_times_array[self.prev_last_point[i]][self.new_last_point[i]]
                    logging.info(f"checking the new label ({self.new_visited},{self.new_last_point},{self.new_time}) with distances {self.new_dist}")
                    self.label_check = {}
                    self.sorted_nlp, self.sorted_nt, self.sorted_nd, self.sorted_vo = zip(*sorted(zip(self.new_last_point, self.new_time, self.new_dist, self.vehicle_order)))
                    self.label_check = {k: v for k, v in self.memo.items() if k[0]==self.new_visited and k[1]==self.sorted_nlp}
                    if not self.duplicate_label_check_VRP():
                        continue
                    
                    #input()
                    if not self.VRP_time_window_check():
                        continue
                    #input()
                    logging.info(f"tests started")
                    if T2:
                        self.before_VRP()
                        if not self.VRP_test2():
                            logging.debug(f"tests ended, label rejected")
                            continue           
                    else:
                        logging.info(f"test 2 is not being used")
                    #input()
                    if T1:                    
                        if not self.VRP_test1A():
                            logging.debug(f"tests ended, label rejected")
                            continue

                    else:
                        logging.info(f"test 1 is not being used")
                    #input()
                    logging.info(f"tests ended")
                    self.VRP_dominance_test_update()
                    #input()
        logging.debug(f"queue = {self.queue}")
        logging.debug(f"memo = {self.memo}")

                
        #input()
        #end while
        self.optimal_path, self.optimal_cost = self.retrace_optimal_path_VRP(self.memo, self.number_of_jobs)
    
        return self.optimal_path, self.optimal_cost, self.df1

    def Solver(self, read_in_data, data, random_data, instances, timeframe, locationframe, servicetime, serviceframe, travel_times_multiplier, save_name, T1, T2, T3):
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
            self.VRP_Solve(T1, T2)
            #self.retrace_optimal_path_VRP(self.memo, self.number_of_jobs )
            self.run_time = round(time.time() - self.t, 3)
        print(f"the memo length is {len(self.memo)}")
        if self.number_of_vehicles == 1:
            print(f"the length of the test 3 rejected labels is {self.test3_rejected_labels}")
        print(f"Found optimal path in {self.run_time} seconds.") 
        print(f"Optimal cost: {self.optimal_cost}, optimal path: {self.optimal_path}")
        print(f"duplicate label check rejected {self.dup_lab_rejected} labels.")
        print(f"time window check rejected {self.TW_rejected} labels.")
        print(f"test 2 rejected {self.test2_rejected} labels.")
        print(f"test 1 rejected {self.test1_rejected} labels.")
        print(f"dominance check rejected {self.dom_lab_rejected} labels.")

        return

    
#logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
a = VRP_Problem(number_of_vehicles = 2)
a.Solver(read_in_data = True, data = 'VRP_testing_1.csv', random_data = False, instances = 6, timeframe = 450, locationframe = 100, servicetime = True, serviceframe = 25, travel_times_multiplier = 1, save_name = 'VRP_small_test_v2.csv', T1 = True, T2 = True, T3 = False)



    
