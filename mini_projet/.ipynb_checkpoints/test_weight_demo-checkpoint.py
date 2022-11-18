import gym
import torch
import torch.nn as nn
import time

env = gym.make("LunarLander-v2")
nb_actions = 4
nb_observations = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device used {device}")


class DuelingQNetwork(nn.Module) :
    
    def __init__(self,
              nb_actions,
              nb_observations) : 
        
        super().__init__()
        self.nb_actions = nb_actions
        self.nb_observations = nb_observations
        
        self.net = nn.Sequential(
            nn.Linear(nb_observations,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32)
        )
        
        self.net_advantage = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32,nb_actions)
        )
        
        self.net_state_value = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32,1)
        )
        
    def advantage(self,x) :
        return self.net_advantage(self.net(x))
    
    def state_value(self,x) :
        return self.net_state_value(self.net(x))
    
    def forward(self,x) :
        return self.state_value(x) + self.advantage(x) - torch.mean(self.advantage(x),dim=1).unsqueeze(1)


q_network = DuelingQNetwork(nb_actions,nb_observations).to(device)

q_network.load_state_dict( torch.load("./best_model_d3qn_lunarlanderdiscret") )

def test(q_network,fps) :
    
    state = env.reset()
    done = False
    cum_sum = 0
    while not done :
        env.render()
        state_t = torch.as_tensor(state , dtype = torch.float32,device = device).unsqueeze(0)
        action = torch.argmax(q_network(state_t)).item()
        new_state,reward,done,_ = env.step(action)
        state = new_state
        cum_sum += reward
        time.sleep(1/fps)
    return cum_sum

result = test(q_network,20)
print(f"resultat : {result}")