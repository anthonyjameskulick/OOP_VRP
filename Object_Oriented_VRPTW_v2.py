
import csv
import time
import numpy as np
np.random.seed(42)
import random
import pandas as pd
import logging

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
        if number_of_vehicles == 1:
            self.prev_last_point = None
            self.prev_time = 0
            self.prev_dist = 0
            self.new_last_point = None
            self.new_time = None
            self.new_dist = None
            self.first = None
        else:
            self.prev_last_point = [0 for i in range(self.number_of_vehicles)] #0 is the depot location
            self.prev_time = [0 for i in range(self.number_of_vehicles)]
            self.prev_dist = [0 for i in range(self.number_of_vehicles)]
            self.new_last_point = [0 for i in range(self.number_of_vehicles)]
            self.new_time = [0 for i in range(self.number_of_vehicles)]
            self.new_dist = [0 for i in range(self.number_of_vehicles)]
            self.first = [0 for i in range(self.number_of_vehicles)]
        
        self.all_points_set = {}
        self.dumas_before_sets = []
        self.VRP_before_set = None
        self.TW = [False for i in range(self.number_of_vehicles)]

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
        return
#random genrator still throwing errors, trying to add a depot creation feature
    def random_data_generator(self, instances, timeframe, locationframe, servicetime, serviceframe, travel_times_multiplier):
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
            start = [0] + random.choices(time0, weights=None, k=instances-1)
            for i in name: self.start_times.append(start[i])
            end=[timeframe]
            for i in name: end.append(random.randrange(start[i]+25,timeframe+25))
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
            testdata = pd.DataFrame({'name' : name, 'xcoord' : x, 'ycoord' : y, 'start' : start, 'end' : end, 'service' : service_time})
            testdata.to_csv('testinstances.csv', sep='\t',index=False)
            self.df = pd.DataFrame({'xcoord' : x, 'ycoord' : y, 'start time' : start, 'end time' : end, 'service time' : service_time})

        return 

    def reset_problem(self):
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
        self.prev_last_point = None
        self.prev_time = None
        self.prev_dist = None
        self.new_visited = []
        self.new_last_point = None
        self.new_time = None
        self.new_dist = None
        self.first = None
        return

    def dumas_latest_departure_time(self, x, y):
        ldt = self.end_times[y]-self.service_times[y]-self.travel_times_array[x][y]-self.service_times[x]
        return ldt

    def dumas_before(self, x):
        for i in self.all_points_set:
                if self.start_times[i]+self.service_times[i]+self.travel_times_array[i][x] > self.end_times[x]:
                    self.dumas_before_sets[i].append(x)
                    logging.debug(self.dumas_before_sets)
                else:
                    continue
        return self.dumas_before_sets
    
    def before_VRP(self):
        Y = [self.dumas_before_sets[x] for x in self.new_last_point]
        self.VRP_before_set = set(Y[0]).intersection(*Y)
        return self.VRP_before_set

    def special_values(self):
        rows, cols = (range(len(self.distances_array)), range(len(self.distances_array)))
        self.LDT_array = np.array([[self.dumas_latest_departure_time(i,j) for i in rows] for j in cols])
        self.before_sets = [self.dumas_before(i)[i] for i in self.all_points_set]
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
            logging.debug(f"NT = {self.new_time[i]}")
            logging.debug(f"ET = {self.end_times[self.new_last_point[i]]}")
            if self.new_time[i] <= self.end_times[self.new_last_point[i]]:
                self.TW[i] = True
                logging.debug(f"TW[{i}] = true")  
                TW = True
            else:
                self.TW[i] = False
                logging.debug(f"TW[{i}] = false")
                logging.info(f"time window check failed")
                TW = False
                break
            logging.debug(f"time window check passed")
        
        return TW

    def dumas_first(self):
        res1 = {k: v for k, v in self.memo.items() if k[0]==self.new_visited and k[1]==self.new_last_point}
        res2 = {value: key for key, value in res1.items()}
        if len(res2) == 0: 
            self.first = self.new_time
        else:
            res3 = min(res2.keys(), key=lambda x: res2[x][2])
            self.first = res2[res3][2]
        return

    def VRP_first(self):
        res1 = {k: v for k, v in self.memo.items() if k[0]==self.new_visited and k[1]==self.new_last_point}
        res2 = {value: key for key, value in res1.items()}
        if len(res2) == 0:
            for i in range(self.number_of_vehicles):
                self.first[i] = self.new_time[i]
        else:
            for i in range(self.number_of_vehicles):
                res3 = min(res2.keys(), key=lambda x: res2[x][2])
                self.first[i] = res2[res3][2][i]
        return

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
        logging.debug(f"before({self.new_last_point})={self.VRP_before_set}")
        logging.debug(f"S = {self.new_visited}")
        if self.VRP_before_set.issubset(set(self.new_visited)) == False:
            test2 = False
            logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) fails test 2")
        else: 
            test2 = True
            logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) passes test 2")
        return test2

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

    def VRP_test1(self):
        if len(self.all_points_set.difference(self.new_visited)) == 0:
            test1 = True
        elif self.first[i] > min(self.LDT_array[j][self.new_last_point] for i in range(self.number_of_vehicles) for j in self.all_points_set.difference(self.new_visited)): 
            logging.debug(f"first = {self.first} > {min(self.LDT_array[j][self.new_last_point] for j in self.all_points_set.difference(self.new_visited))}")
            logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) fails test 1")
            test1 = False
        else:
            logging.debug(f"first = {self.first} > {min(self.LDT_array[j][self.new_last_point] for j in self.all_points_set.difference(self.new_visited))}")
            logging.info(f"({self.new_visited}, {self.new_last_point}, {self.new_time}) passes test 1")
            test1 = True
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
            #self.optimal_path = optimal_path
            #self.optimal_cost = optimal_path
            self.df1 = pd.DataFrame({'opt path': optimal_path, 'start': start1, 'arrival': optimal_path_arrival_times, 'departure': optimal_path_departure_times, 'end': end1 })
        return optimal_path, optimal_cost

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
                    if not self.before_VRP():
                        self.VRP_test2()
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
        self.dumas_before_sets = [[ ] for y in self.all_points_set]
        self.special_values()
        self.queue = [(tuple([0]), tuple(self.prev_last_point), tuple(self.prev_time))] #job 0 is the designated depot
        logging.debug(self.queue)
        self.memo [(tuple([0]), tuple(self.prev_last_point), tuple(self.prev_time))] = (tuple([self.prev_dist, self.prev_last_point, self.prev_time])) #job 0 is the designated depot
        logging.debug(self.memo)
        while self.queue: 
            self.prev_visited, self.prev_last_point, self.prev_time = self.queue.pop(0)
            self.prev_dist, _, _ = self.memo[(self.prev_visited, self.prev_last_point, self.prev_time)]
            logging.debug(self.prev_visited)
            logging.debug(self.prev_last_point)
            logging.debug(self.prev_time)
            logging.debug(self.prev_dist)
            to_visit = self.all_points_set.difference(set(self.prev_visited))
            logging.debug(to_visit)
            for i in range(self.number_of_vehicles):
                logging.info(f"for vehicle {i}")
                self.new_last_point[i-1] = a.prev_last_point[i-1]
                self.new_dist[i-1] = self.prev_dist[i-1]
                self.new_time[i-1] = self.prev_time[i-1]
                for self.new_last_point[i] in to_visit:    
                    logging.info(f"visiting job {self.new_last_point[i]}")
                    self.new_visited = tuple(sorted(list(self.prev_visited) + [self.new_last_point[i]]))
                    self.new_dist[i] = self.prev_dist[i] + self.distances_array[self.prev_last_point[i]][self.new_last_point[i]]
                    self.new_time[i] = max(self.prev_time[i], self.start_times[self.prev_last_point[i]]) + self.service_times[a.prev_last_point[i]] + self.travel_times_array[self.prev_last_point[i]][self.new_last_point[i]]
                    self.new_last_point, self.new_dist, self.new_time = zip(*sorted(zip(self.new_last_point, self.new_dist, self.new_time)))
                    logging.info(f"checking the new label ({self.new_visited},{self.new_last_point},{self.new_time}) with distances {self.new_dist}")
                    if not self.VRP_time_window_check():
                        continue #this continue will send you back to the top of the for loop
                    logging.info(f"tests started")      
                    if T2:
                        if not self.before_VRP():
                            self.VRP_test2()
                            logging.info(f"tests ended")
                            continue
                    else:
                        logging.info(f"test 2 is not being used")
                
                    self.VRP_first()
                    if T1:                    
                        if not self.VRP_test1():
                            logging.info(f"tests ended")
                            continue
                    else:
                        logging.info(f"test 1 is not being used")
                    
                    logging.info(f"tests ended")

                    #self.dominance_test()
                

        #end while
        #self.optimal_path, self.optimal_cost = self.retrace_optimal_path_TSPTW(self.memo, self.number_of_jobs)
    
        return #self.optimal_path, self.optimal_cost, self.df1

    def Solver(self, read_in_data, data, random_data, instances, timeframe, locationframe, servicetime, serviceframe, T1, T2, T3):
        self.reset_problem()
        if read_in_data == True:
            self.read_in_data(data, 1)
            logging.info(f" the problem is {self.df}")
        else:
            logging.debug('no read in data given')
        if random_data == True:
            self.random_data_generator(instances, timeframe, locationframe, servicetime, serviceframe, 1)
            print(self.df)
        else:
            logging.debug('no random data generated')
        if self.number_of_vehicles == 1:
            logging.info(f"TSP situation")
            self.t = time.time()
            self.Dumas_TSPTW_Solve(T1, T2, T3)
            self.run_time = round(time.time() - self.t, 3)
            logging.debug(self.before_sets)
            logging.debug(self.LDT_array)
        else:
            logging.debug(f"VRP situation")
            self.t = time.time()
            self.VRP_Solve(T1, T2, T3)
            self.run_time = round(time.time() - self.t, 3)
            logging.debug(self.before_sets)
            logging.debug(self.LDT_array)
        print("the memo length is:")
        print(len(self.memo))
        print("the length of the test 3 rejected labels is:")
        print(len(self.test3_restricted_labels))
        print(f"Found optimal path in {self.run_time} seconds.") 
        print(f"Optimal cost: {self.optimal_cost}, optimal path: {self.optimal_path}")
        print("Time check:")
        print(self.df1)
        return


    
                    
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
a = VRP_Problem(number_of_vehicles = 3)
a.read_in_data('testdata_VRP.csv', travel_times_multiplier = 2)
logging.info(a.df)
a.queue = [(tuple([0]), tuple(a.prev_last_point), tuple(a.prev_time))]
logging.debug(a.queue)
a.memo [(tuple([0]), tuple(a.prev_last_point), tuple(a.prev_time))] = (tuple([a.new_dist, a.prev_last_point, a.prev_time, a.prev_dist])) 

while a.queue:
    logging.debug(a.memo)
    a.prev_visited, a.prev_last_point,_ = a.queue.pop(0)
    to_visit = a.all_points_set.difference(set(a.prev_visited))
    logging.debug(to_visit)
    for i in range(a.number_of_vehicles):
        logging.info(f"for vehicle {i}")
        a.new_last_point[i-1] = a.prev_last_point[i-1]
        
        for a.new_last_point[i] in to_visit:
            _, _, prev_time1, prev_dist1 = a.memo[(tuple(a.prev_visited), tuple(a.prev_last_point), tuple(a.prev_time))]
            logging.debug(f"prev_dist = {a.prev_dist}, prev_time = {a.prev_time}")
            a.new_dist = new_dist1 = prev_dist1
            logging.debug(f"new_dist = {a.new_dist}")
            a.new_time = new_time1 = prev_time1
            logging.debug(f"new_time = {a.new_time}")
            logging.info(f"visiting job {a.new_last_point[i]}")
            a.new_visited = tuple(sorted(list(a.prev_visited) + [a.new_last_point[i]]))
            new_dist1[i] = a.new_dist[i] + a.distances_array[a.prev_last_point[i]][a.new_last_point[i]]
            logging.debug(f"new_dist = {new_dist1}")
            new_time1[i] = max(prev_time1[i], a.start_times[a.prev_last_point[i]]) + a.service_times[a.prev_last_point[i]] + a.travel_times_array[a.prev_last_point[i]][a.new_last_point[i]]
            logging.debug(f"new_time = {new_time1}")
            #temp1, temp2, temp3 = (list(t) for t in zip(*sorted(zip(list(a.new_last_point), list(a.new_time1), list(a.new_dist1)))))
            logging.info(f"checking the new label ({a.new_visited},{a.new_last_point},{new_time1}) with distances {new_dist1}")
            if a.VRP_time_window_check():
                a.VRP_first()
                logging.info(f"first = {a.first}")
                temp1, temp2, temp3, temp4, temp5, temp6 = zip(*sorted(zip(a.new_last_point, new_time1, new_dist1, a.prev_last_point, a.prev_time, a.prev_dist))) 
                #if len({k: v for k, v in a.memo.items() if k[0]==a.new_visited and k[1]==temp1}) != 0:
                a.memo[(tuple(a.new_visited), tuple(a.new_last_point), tuple(new_time1))]=tuple((new_dist1, a.prev_last_point, a.prev_time, a.prev_dist))
                a.queue += [(tuple(a.new_visited), tuple(a.new_last_point), tuple(new_time1))]
                logging.info(f"added the label ({a.new_visited}, {a.new_last_point}, {new_time1}) with distances {new_time1}")
                input()
                #else:
                    #logging.info(f"rejected (duplicate) the label ({a.new_visited}, {a.new_last_point}, {new_time1}) with distances {new_time1}")
            else:
                logging.info(f"rejected the label ({a.new_visited}, {a.new_last_point}, {new_time1}) with distances {new_time1} for TW failure")
                continue

            