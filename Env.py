# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


""" 
Locations:        m =>  {A:1, B:2, C:3, D:4, E:5} 
Time of the Date: t =>  {24 hours clock 00:00, 01:00 ....23:00, represented by integers 0,1,2...23}
Day of the Week:  d =>  {Monday, Tuesday...Sunday, represented as 0,1,2...6}
"""
class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        # Define the accumulative travel hours for the Cab driver
        self.accumulative_travel_hrs=0
        
        # Possible action-state spcae = ((m-1)*m) + 1 = ((5-1)*5)+ 1 = 21 (On state to go Offline)
        self.action_space = [(1,2),(1,3),(1,4),(1,5),
                             (2,1),(2,3),(2,4),(2,5),
                             (3,1),(3,2),(3,4),(3,5),
                             (4,1),(4,2),(4,4),(4,5),
                             (5,1),(5,2),(5,3),(5,4),
                             (0,0)]
        # Total possible states (Xi Tj Dk) = 1...m,1...t,1...d
        self.state_space = [(p, q, r) for p in range(1, m+1)
                                      for q in range(t)
                                      for r in range(d)]
        # Initialize state with random state space
        self.state_init = random.choice([(1,0,0),(2,0,0),(3,0,0),(4,0,0),(5,0,0)])

        # Start the first round
        self.reset()
        


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. 
        This method converts a given state into a vector format. 
        Hint: The vector is of size m + t + d."""
        if not state:
            return
        state_encod = [0] * (m + t + d)

        #Encode the location
        state_encod[state[0] - 1] = 1
        
        #Encode Hour of the Day 
        state_encod[m + state[1]] = 1

        #Encode Day of the Week
        state_encod[m + t + state[2]] = 1
        return state_encod


    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""
        if not state:
            return
        state_encod = [0] * (m + t + d + m + m)
        # Encode Location
        state_encod[state[0] - 1] = 1
        #Encode Hour of the Dat 
        state_encod[m + state[1]] = 1
        #Encode Day of the Week
        state_encod[m + t + state[2]] = 1

        # actions to be represented as location to location 
        if action[0] and action[1]:
            state_encod[(m+t+d) + action[0] -1] = 1
            state_encod[(m+t+d+m) + action[1] - 1] = 1
        return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 1:
            requests = np.random.poisson(2)
        if location == 2:
            requests = np.random.poisson(12)
        if location == 3:
            requests = np.random.poisson(4)
        if location == 4:
            requests = np.random.poisson(7)
        if location == 5:
            requests = np.random.poisson(8)

        # Limit the number of request to 15
        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        if (0,0) not in actions:
            actions.append([0,0])
            possible_actions_index.append(20)

        return possible_actions_index,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward
        reward = (revenue earned from pick up point p to drop point q) - 
                 (Cost of fuel used in moving from p to q) -
                 (Cost of fuel used in moving point i to p)
        """
        cur_loc = state[0]
        start_loc = action[0]
        end_loc = action[1]
        time_of_day = state[1]
        day_of_week = state[2]

        # Calculates new time and day
        def get_new_time_of_day(time_of_day,day_of_week, total_time):
            time_of_day = time_of_day + total_time % (t - 1)
            day_of_week = day_of_week + (total_time // (t - 1))

            if time_of_day > (t - 1):
                day_of_week = day_of_week + (time_of_day // (t - 1))
                time_of_day = time_of_day % (t - 1)
                if day_of_week > (d - 1):
                    day_of_week = day_of_week % (d - 1)
            return time_of_day, day_of_week

        # Calculates total travel time
        def get_total_travel_time(cur_loc, start_loc, end_loc, time_of_day, day_of_week):
            if not start_loc and not end_loc:
                return 0, 1
            
            time_1 = 0
            if start_loc and cur_loc != start_loc:
                time1 = int(Time_matrix[cur_loc-1][start_loc-1][time_of_day][day_of_week])

                # Compute new time_of_day & day_of_week after travel time1
                time_of_day, day_of_week = get_new_time_of_day(time_of_day,day_of_week, time_1)

            time_2 = int(Time_matrix[start_loc-1][end_loc-1][time_of_day][day_of_week])
            #print("time_1={} time_2={}".format(time_1,time_2))

            return time_1,time_2

        #
        time_1, time_2 = get_total_travel_time(cur_loc,start_loc,end_loc,time_of_day,day_of_week)
        #print("time_1={} time_2={}".format(time_1,time_2))

        if not start_loc and not end_loc:
            reward = -C
        else:
            reward = R * time_2 - C * (time_1 + time_2)

        return reward


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        cur_loc = state[0]
        start_loc = action[0]
        end_loc = action[1]
        time_of_day = state[1]
        day_of_week = state[2]

        #
        def get_total_travel_time(cur_loc,start_loc,end_loc,time_of_day,day_of_week):
            # calculate total time of travel
            if not start_loc and not end_loc:
                return 1

            time_1 = 0
            if start_loc and cur_loc != start_loc:
                time_1 = int(Time_matrix[cur_loc-1][start_loc-1][time_of_day][day_of_week])
                #compute new time_of_day and day_of_week after travel time_1
                time_of_day, day_of_week = get_new_time_of_day(time_of_day,day_of_week, time_1)
            time_2 =  int(Time_matrix[start_loc-1][end_loc-1][time_of_day][day_of_week])
            return time_1+time_2

        #
        def get_new_time_of_day(time_of_day,day_of_week,total_time):
            # calculate new and day
            time_of_day = time_of_day + total_time % (t - 1)
            day_of_week = day_of_week + (total_time // (t - 1))

            if time_of_day > (t - 1):
                day_of_week = day_of_week + (time_of_day // (t - 1))
                time_of_day = time_of_day % (t - 1)
                if day_of_week > (d - 1):
                    day_of_week = day_of_week % (d - 1)
            return time_of_day, day_of_week
        #
        total_travel_time = get_total_travel_time(cur_loc, start_loc, end_loc, time_of_day, day_of_week)
        self.accumulative_travel_hrs += total_travel_time
        new_time_of_day, new_day_of_week = get_new_time_of_day(time_of_day, day_of_week, total_travel_time)

        if not start_loc and not end_loc:
            new_loc = state[0]
        else:
            new_loc = action[1]

        next_state =  (new_loc,new_time_of_day,new_day_of_week)
        return next_state



            # Reset the travel hours
    def reset(self):
        self.accumulative_travel_hrs = 0 
        self.state_init = random.choice([(1,0,0),(3,0,0),(3,0,0),(4,0,0),(5,0,0)])
        return self.action_space, self.state_space, self.state_init


    # Function can be used to test the Environment
    def test_run(self):
        import operator
        Time_matrix = np.load('TM.npy')
        print('Current State: {}'.format(self.state_init))
        
        # Check the request at the init state
        requests = self.requests(self.state_init)
        print('Request: {}'.format(requests))
        
        # Reward computation
        rewards = []
        for req in requests[1]:
            r = self.reward_func(self.state_init,req, Time_matrix)
            rewards.append(r)
        print('Rewards: {}'.format(rewards))
        
        # New Possible States
        new_states = []
        for req in requests[1]:
            s = self.next_state_func(self.state_init, req, Time_matrix)
            new_states.append(s)
        print('New Possible States: {}'.format(new_states))
        
        # Decide the new state based on Max reward
        index, max_reward = max(enumerate(rewards), key=operator.itemgetter(1))
        self.state_init = new_states[index]
        
        print("Maximum Reward : {}".format(max_reward))
        print("Action : {}".format(requests[1][index]))
        print("Total Travel Hours : {}".format(self.accumulative_travel_hrs))
        print("New State : {}".format(self.state_init))
        print("NN Input Layer (Arch#1) : {}".format(self.state_encod_arch1(self.state_init)))
        print("NN Input Layer (Arch#2) : {}".format(self.state_encod_arch2(self.state_init, requests[1][index])))