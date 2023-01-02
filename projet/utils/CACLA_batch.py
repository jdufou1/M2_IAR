"""
Batch CACLA : Continuous Actor Critic Learning Automaton
Implementation and reproduction of the CACLA algorithm used in the paper CACLA 
src : https://dspace.library.uu.nl/bitstream/handle/1874/25514/wiering_07_reinforcementlearning.pdf
developed by : Jérémy DUFOURMANTELLE and Ethan ABITBOL
"""
import numpy as np
import copy
import random
import torch
import torch.nn as nn

from collections import deque

class CACLAbatch() :
    
    def __init__(
        self,
        learning_rate_critic : float,
        learning_rate_actor : float,
        discount_factor : float,
        epsilon : float,
        epsilon_min : float,
        epsilon_decay : float,
        sigma : float,
        nb_episode : int,
        nb_tests : int,
        test_frequency : int,
        batch_size : int,
        size_replay_buffer : int,
        env,
        actor_network,
        critic_network,
        exploration_strategy : str = "gaussian",
        verbose_mode : bool = True
    ) :
        
        self.learning_rate_critic = learning_rate_critic
        self.learning_rate_actor = learning_rate_actor
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.sigma = np.zeros(env.action_space) + sigma
        self.nb_episode = nb_episode
        self.nb_tests = nb_tests
        self.test_frequency = test_frequency
        self.batch_size = batch_size
        self.size_replay_buffer = size_replay_buffer
        self.env = env
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.exploration_strategy = exploration_strategy
        self.verbose_mode = verbose_mode
        
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        self.optimizer_actor= torch.optim.Adam(self.actor_network.parameters(),lr=self.learning_rate_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic_network.parameters(),lr=self.learning_rate_critic)
        
        self.best_model = copy.deepcopy(self.actor_network)
        self.best_value = -1e10
        self.iteration = 0

        self.replay_buffer = deque(maxlen=self.size_replay_buffer)

        
    def learning(self) : 
        
        self.list_rewards_mean = list()
        self.list_rewards_std = list()
        self.list_iteration = list()
        
        for episode in range(self.nb_episode) :
        
            state = self.env.reset()

            done = False

            while not done :
                
                self.iteration += 1
            
                state_t = torch.as_tensor(state , dtype=torch.float32)
                
                action = self.get_action(state_t)
                
                new_state, reward, done = self.env.step(action)
                
                new_state_t = torch.as_tensor(new_state , dtype=torch.float32)
        
                reward_t = torch.as_tensor(reward , dtype=torch.float32)
            
                with torch.no_grad():
                    td_error = (reward_t + 
                                (self.discount_factor * 
                                           (1 - done) * 
                                           self.critic_network(new_state_t)
                                ) 
                                - self.critic_network(state_t))
                
                update = len(self.replay_buffer) >= self.batch_size 
                if update:
                    
                    transitions = random.sample(self.replay_buffer , self.batch_size)

                    states_t = torch.stack([t[0] for t in transitions])
                    td_errors = torch.stack([t[5] for t in transitions])
                    actions_t = torch.stack([t[2] for t in transitions])


                    # learning critic
                    loss_critic = - td_errors.detach() * self.critic_network(states_t)

                    self.optimizer_critic.zero_grad()
                    loss_critic.backward()
                    self.optimizer_critic.step()
                
                if td_error > 0 :
                    
                    action_t = torch.as_tensor(action , dtype=torch.float32)

                    transition = (state_t , reward, action_t , new_state_t, done, td_error)
                    self.replay_buffer.append(transition)

                    if update : 

                        # learning actor
                        loss_actor = - ( (actions_t - self.actor_network(states_t).detach()) * self.actor_network(states_t) ).mean()

                        self.optimizer_actor.zero_grad()
                        loss_actor.backward()
                        self.optimizer_actor.step()
                
                state = new_state
            
            # testing
            
            if episode % self.test_frequency == 0 :
                rewards_tests = list()
                for t in range(self.nb_tests) :
                    rewards_tests.append(self.test())
                rewards_tests = np.array(rewards_tests)
                    
                self.list_rewards_mean.append(rewards_tests.mean())
                self.list_rewards_std.append(rewards_tests.std())
                self.list_iteration.append(self.iteration)
                
                if self.verbose_mode :
                    print(f"{episode}/{self.nb_episode} - iteration : {self.iteration} - rewards value test : {rewards_tests.mean()} - best value : {self.best_value}")
                
                if self.best_value < rewards_tests.mean() :
                    self.best_model.load_state_dict(self.actor_network.state_dict())
                    self.best_value = rewards_tests.mean()
                    
    
    def get_action(self,state_t) :
        if self.exploration_strategy == "gaussian" :
            return torch.as_tensor(
                        np.array(
                            np.random.normal(loc=self.actor_network(state_t).detach().numpy(),
                            scale=self.sigma,size=(1,self.action_space))
                        ),
                        dtype=torch.float32
                )[0].detach().numpy()
        elif self.exploration_strategy == "egreedy" :
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            if np.random.rand() > self.epsilon :
                return self.actor_network(state_t).detach().numpy()
            else :
                return torch.as_tensor(
                        np.array(
                            np.random.normal(loc=self.actor_network(state_t).detach().numpy(),
                            scale=self.sigma,size=(1,self.action_space))
                        ),
                        dtype=torch.float32
                )[0].detach().numpy()
        else :
            raise Exception("The exploration strategy must be gaussian or egreedy")
    
    
        
    def test(self) :  
        
        list_rewards = list()
        state = self.env.reset()
        done = False
        while not done :
            state_t = torch.as_tensor(state , dtype=torch.float32)
            action =  self.actor_network(state_t).detach().numpy()
            new_state, reward, done = self.env.step(action)
            list_rewards.append(reward)
            state = new_state
        return sum(list_rewards) / len(list_rewards)