import argparse
from itertools import count

import os, sys, random
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from CartPoleContinuous import ContinuousCartPoleEnv
import matplotlib.pyplot as plt

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''













parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
# parser.add_argument("--env_name", default="Pendulum-v0")
parser.add_argument("--env_name", default="CartPoleContinuous")
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=100000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=200, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
# env = gym.make(args.env_name)
env = ContinuousCartPoleEnv()
if args.seed:
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = env.observation_space
action_dim = env.action_space
max_action = float(1)
min_Val = torch.tensor(1e-7).float().to(device) # min value

directory = './exp' + script_name + args.env_name +'./'

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action * torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):
        list_td_errors = list()

        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            td_error = target_Q - current_Q
            list_td_errors.append(td_error.cpu().detach().numpy().sum())
            
            mask = (td_error > 0).squeeze()
            
            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()


            # if sum(mask) > 0 :
            # Compute actor loss
            actor_loss = -self.critic(state[mask], self.actor(state[mask])).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1
        return list_td_errors

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

def main():
    
    
    nb_training = 10
    list_td_errors = list()
    list_rewards = list()
    list_rewards_mean = list()
    list_rewards_std = list()
    for m in range(nb_training) :
        print(f"Iteration training : {m}")
        
        tmp_rewards = list()
        tmp_errors = list()
        
        agent = DDPG(state_dim, action_dim, max_action)
        ep_r = 0
        if args.mode == 'test':
            agent.load()
            for i in range(args.test_iteration):
                state = env.reset()
                for t in count():
                    action = agent.select_action(state)
                    next_state, reward, done = env.step(np.float32(action))
                    ep_r += reward
                    # env.render()
                    if done or t >= args.max_length_of_trajectory:
                        print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                        ep_r = 0
                        break
                    state = next_state

        elif args.mode == 'train':
            writer = SummaryWriter(directory)
            if args.load: agent.load()
            total_step = 0
            for i in range(args.max_episode):
                total_reward = 0
                step =0
                state = env.reset()
                for t in count():
                    action = agent.select_action(state)
                    action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space)).clip(
                        -1, 1)

                    next_state, reward, done = env.step(action)
                    #if args.render and i >= args.render_interval :
                    agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))

                    state = next_state
                    if done:
                        break
                    step += 1
                    total_reward += reward
                total_step += step+1
                print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
                writer.add_scalar('Reward', total_reward, global_step=i)
                tmp_rewards.append(total_reward)
                res_errors = agent.update()
                tmp_errors.append(sum(res_errors) / len(res_errors))
                # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

                if i % args.log_interval == 0:
                    agent.save()
                    
                if i >= 100 : # on arrete l'entrainement au 100ieme episode
                    break 
            
            list_rewards.append(tmp_rewards)
            list_td_errors.append(tmp_errors)
            """
            tmp_rewards = np.array(tmp_rewards)
            list_rewards_mean.append(tmp_rewards)
            list_rewards_std.append(tmp_rewards.std())
            """
            
    list_rewards = np.array(list_rewards)
    list_rewards_mean = list_rewards.mean(axis = 0)
    list_rewards_std = list_rewards.std(axis = 0)
    
    list_td_errors = np.array(list_td_errors)
    list_td_errors_mean = list_td_errors.mean(axis = 0)
    
    plt.figure()
    sum_mean1 = np.array(list_rewards_mean) + np.array(list_rewards_std)
    sum_mean2 = np.array(list_rewards_mean) - np.array(list_rewards_std)
    plt.fill_between(np.arange(0,len(list_rewards_mean),1),sum_mean1,sum_mean2, color = 'mistyrose',label = 'std reward')
    plt.plot(np.arange(0,len(list_rewards_mean),1),list_rewards_mean, c= 'r', label = 'mean_reward')
    plt.legend()
    plt.xlabel('episode')
    plt.ylabel('rewards')
    plt.title('variance and mean rewards for DDPG avec TDerror > 0')
    plt.show()
    
    plt.figure()
    plt.plot(list_td_errors_mean)
    plt.xlabel("episode")
    plt.ylabel("TD error")
    plt.title("Evolution des TD errors au cours du temps pour DDPG avec TDerror > 0")
    plt.show()
            

if __name__ == '__main__':
    main()