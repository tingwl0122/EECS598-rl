"""
A sample agent
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import gym
import os
import time


# Actor Neural Network
class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc_units=512, fc1_units=512):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc1_units)
        self.fc3 = nn.Linear(fc1_units, action_size)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.torch.tanh(self.fc3(x))

# Agent Class
class Agent:
    def __init__(
        self,
        gamma = 0.99, #discount factor
        lr_actor = 3e-4,
        lr_critic = 3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = gym.make('BipedalWalkerHardcore-v3')
        self.create_actor()

    def create_actor(self):
        params = {
            'state_size':      self.env.observation_space.shape[0],
            'action_size':     self.env.action_space.shape[0],
            'seed':            88
        }
        self.actor = Actor(**params).to(self.device)
        self.actor.load_state_dict(torch.load('actor_512.pth', map_location=torch.device('cpu'))) #a pre saved model

    def select_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(state).cpu().data.numpy()
        return np.squeeze(actions)
