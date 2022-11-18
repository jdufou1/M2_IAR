"""
Implementation of the Tracking environment used in the paper CACLA 
src :
developed by : Jérémy DUFOURMANTELLE and Ethan ABITBOL

"""

import numpy as np

class Tracking :
    
    def __init__(self) :
        """
        using exemple : 
            env = Tracking() # to create the environment
            initial_state = env.reset() # to initialize the environment and get the first state
            new_state,reward,done = env.step(action) # to make an iteration in the environment
        """ 
        self.min_val = 0.0
        self.max_val = 10.0
        self.step_val = np.array([0.161,0.161])
        self.max_iteration = 300
        self.observation_space = 4
        self.action_space = 2
        
    def step(self,action) :
        """
        action : vecteur deux dimensions a atteindre, on va donc calculer le vecteur directeur
                 entre l'agent et l'objectif
        """
        self.iteration += 1
        vector_director = action - self.agent
        vector_director_normalized = np.linalg.norm(vector_director)
        future_agent_position = self.agent + (vector_director/vector_director_normalized) * self.step_val
        
        if not self.inside_box(future_agent_position) and not self.outside(future_agent_position):
            self.agent = future_agent_position
        
        self.target_angle_value += self.step_val
        self.target = self.Middle + np.array([self.Radius * np.cos(self.target_angle_value[0]),self.Radius * np.sin(self.target_angle_value[1])])
        
        squarred_distance = (self.agent[0] - self.target[0])**2 + (self.agent[1] - self.target[1])**2
        
        success = (squarred_distance < 0.1)
        
        """
        if success :
            reward = (self.max_iteration - self.iteration)
        else :
            reward = - euclidian_distance
        """ 
        reward = - squarred_distance # dans le papier, ils utilisent uniquement cette valeur
        
        return np.concatenate((self.agent,self.target)), reward, ( success or self.iteration >= self.max_iteration)
    
    def reset(self) :
        """
        return a 4-dimension vector which is the concatenation between target and agent position
        """
        self.iteration = 0
        self.agent = np.array([5.0,5.0])
        self.target = np.array([1.0,4.5])
        self.target_angle_value = np.array([0.0,0.0])
        
        self.Middle = np.array([4.5,4.5])
        self.Radius = 4
        
        return np.concatenate((self.agent,self.target))
    
    def render(self) :
        """
        TODO : display the environment's elements
        """
        pass
    
    def outside(self,position) :
        """
        return boolean if the position is outside from the environment
        """
        return position[0] > self.max_val or position[0] < self.min_val or position[1] > self.max_val or position[1] < self.min_val
    
    def inside_box(self,position) :
        """
        return boolean if the position is inside the environment's box
        """
        return position[0] > 4 and position[0] < 9 and position[1] > 5 and position[1] < 6