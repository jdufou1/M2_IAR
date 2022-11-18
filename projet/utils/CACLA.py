"""
CACLA : Continuous Actor Critic
Implementation and reproduction of the CACLA algorithm used in the paper CACLA 
src : https://dspace.library.uu.nl/bitstream/handle/1874/25514/wiering_07_reinforcementlearning.pdf
developed by : Jérémy DUFOURMANTELLE and Ethan ABITBOL
"""
import numpy as np
import copy
import torch
import torch.nn as nn

class CACLA() :
    
    def __init__(
        self,
        learning_rate_critic : float,
        learning_rate_actor : float,
        discount_factor : float,
        sigma : float,
        nb_episode : int,
        test_frequency : int,
        env,
        actor_network,
        critic_network,
        verbose_mode : bool = True
    ) :
        
        self.learning_rate_critic = learning_rate_critic
        self.learning_rate_actor = learning_rate_actor
        self.discount_factor = discount_factor
        self.sigma = np.zeros(env.action_space) + sigma
        self.nb_episode = nb_episode
        self.test_frequency = test_frequency
        self.env = env
        self.actor_network = actor_network
        self.critic_network = critic_network
        self.verbose_mode = verbose_mode
        
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
        self.optimizer_actor= torch.optim.Adam(self.actor_network.parameters(),lr=self.learning_rate_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic_network.parameters(),lr=self.learning_rate_critic)
        
        self.best_model = copy.deepcopy(self.actor_network)
        self.best_value = -1e10
        self.iteration = 0
        
    def learning(self) : 
        
        self.list_rewards = list()
        
        for episode in range(self.nb_episode) :
        
            state = self.env.reset()

            done = False

            while not done :
                
                self.iteration += 1
            
                state_t = torch.as_tensor(state , dtype=torch.float32)
                
                action = torch.as_tensor(
                    np.array(
                        np.random.normal(loc=self.actor_network(state_t).detach().numpy(),
                        scale=self.sigma,size=(1,self.action_space))
                    ),
                    dtype=torch.float32
                )[0].detach().numpy() # gaussian exploration
                
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
                
                # learning critic
                loss_critic = - td_error.detach() * self.critic_network(state_t)

                self.optimizer_critic.zero_grad()
                loss_critic.backward()
                self.optimizer_critic.step()
                
                if td_error > 0 :
                    
                    action_t = torch.as_tensor(action , dtype=torch.float32)

                    # learning actor
                    loss_actor = - ( (action_t - self.actor_network(state_t).detach()) * self.actor_network(state_t) ).mean()

                    self.optimizer_actor.zero_grad()
                    loss_actor.backward()
                    self.optimizer_actor.step()
                
                state = new_state
            
            # testing
            
            if episode % self.test_frequency == 0 :
                rewards_test = self.test()
                self.list_rewards.append(rewards_test)
                
                if self.verbose_mode :
                    print(f"{episode}/{self.nb_episode} - iteration : {self.iteration} - rewards value test : {rewards_test} - best value : {self.best_value}")
                
                if self.best_value < rewards_test :
                    self.best_model.load_state_dict(self.actor_network.state_dict())
                    self.best_value = rewards_test
                    
                
        
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