import torch
import torch.nn as nn

class CriticNetwork(nn.Module) :
    
    def __init__(
        self,
        nb_neurons : int,
        observation_space : int
    ) :
        
        super().__init__()
        
        self.nb_neurons = nb_neurons
        self.observation_space = observation_space
        
        self.net = nn.Sequential(
            nn.Linear(self.observation_space, self.nb_neurons),
            nn.Sigmoid(),
            nn.Linear(self.nb_neurons, 1)
        )
        
    def forward(self,x) :
        return self.net(x)