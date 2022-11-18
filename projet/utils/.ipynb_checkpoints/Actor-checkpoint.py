import torch
import torch.nn as nn

class ActorNetwork(nn.Module) :
    
    def __init__(
        self,
        nb_neurons : int,
        action_space : int,
        observation_space : int
    ) :
        
        super().__init__()
        
        self.nb_neurons = nb_neurons
        self.action_space = action_space
        self.observation_space = observation_space
        
        self.net = nn.Sequential(
            nn.Linear(self.observation_space, self.nb_neurons),
            nn.Sigmoid(),
            nn.Linear(self.nb_neurons, self.action_space)
        )
        
    def forward(self,x) :
        return self.net(x)