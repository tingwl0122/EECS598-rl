# -*- coding: utf-8 -*-
# ``agent.py``

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import torch.optim as optim
from Arena import Arena
from OthelloGame import OthelloGame
from OthelloPlayers import RandomPlayer, HumanOthelloPlayer, GreedyOthelloPlayer
import math
from tqdm.notebook import tqdm
from random import shuffle


# Actor Neural Network
class PolicyNet(nn.Module):

    def __init__(self, game):
        super().__init__()
        
        # parameters
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.num_channels = 256  # number of channels for the Conv2d layer
        self.dropout = 0.3  # Dropout probability
        
        # convolutional layers
        self.conv1 = nn.Conv2d(1, self.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(self.num_channels)
        self.bn2 = nn.BatchNorm2d(self.num_channels)
        self.bn3 = nn.BatchNorm2d(self.num_channels)

        self.fc1 = nn.Linear(self.num_channels*(self.board_x-2)*(self.board_y-2), 512)
        self.fc_bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, self.action_size)

        self.fc3 = nn.Linear(512, 1)

    def forward(self, s):
        
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = s.view(-1, self.num_channels*(self.board_x-2)*(self.board_y-2))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 512

        # log probability of actions in state s
        pi = F.log_softmax(self.fc2(s), dim=1)                                                   # batch_size x action_size
        # value of state s
        v = torch.tanh(self.fc3(s))                                                              # batch_size x 1

        return pi, v


class MCTS:
    
    def __init__(self, game, policy_net):
        self.game = game
        self.policy_net = policy_net

        self.num_MCTS_sims = 50  # number of simulations for MCTS for each action
        self.bonus_term_factor = 1.0

        self.Qsa = {}  # stores Q values for s,a
        self.Nsa = {}  # stores number of times edge s,a was visited
        self.Ns = {}  # stores number of times board s was visited
        self.Ps = {}  # stores initial policy (returned by policy network)

        self.Es = {}  # stores game.getGameEnded for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        #self.device = "cuda"
    def search(self, canonicalBoard):
    
        # Use string representation for the state
        s = self.game.stringRepresentation(canonicalBoard)
        
        # Update self.Es
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(canonicalBoard, 1)
        
        
        if self.Es[s] != 0:  # The game ended, which means that s is a terminal node
            # If the current player won, then return -1 (The value for the other player).
            # Otherwise, return 1 (The value for the other player).
            return -self.Es[s]

        if s not in self.Ps:  # There is no policy for the current state s, which means that s is a leaf node (a new state)
            
            #self.Ps = {}  # stores initial policy (returned by policy network)

            # Set Q(s,a)=0 and N(s,a)=0 for all a
            for a in range(self.game.getActionSize()):
                self.Qsa[(s, a)] = 0
                self.Nsa[(s, a)] = 0
            
            # Calculate the output of the policy network, which are the policy and the value for state s

            # the numpy representation of board converted to torch tensor
            board = torch.FloatTensor(canonicalBoard.astype(np.float64),).view(1, self.policy_net.board_x,
                                                                              self.policy_net.board_y)
            #board = board.to(self.device)
            #print(board.device)
            self.policy_net.eval()

            # get two output of the policy network regarding to the newly seen state
            with torch.no_grad():
                pi, v = self.policy_net(board)
            self.Ps[s] = torch.exp(pi).data.cpu().numpy()[0]  # The policy for state s
            v = v.data.cpu().numpy()[0][0]  # The value of state s
            
            # Masking invalid moves
            valids = self.game.getValidMoves(canonicalBoard, 1)
            self.Ps[s] = self.Ps[s] * valids
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            
            self.Vs[s] = valids  # Stores the valid moves
            self.Ns[s] = 0
            return -v
    
        # pick the action with the highest upper confidence bound (ucb) and assign it to best_act
        best_act = -1
        valids = self.Vs[s]
        cur_best = -float('inf')
        for a in range(self.game.getActionSize()):
            if valids[a]:
                
                # You can uncomment the following codes and fill in the blanks
                ### BEGIN SOLUTION
                # YOUR CODE HERE

                # compute q-value plus UCB bonus
                Q_val = self.Qsa[(s, a)]
                UCB = self.bonus_term_factor * self.Ps[s][a] * math.sqrt(self.Ns[s]) / ( 1 + self.Nsa[(s, a)] )
                if (Q_val + UCB) > cur_best:
                  cur_best = Q_val + UCB
                  best_act = a
                ### END SOLUTION
        
        # Continue the simulation: take action best_act in the simulation
        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)  # This returns the value for the current player
       
        # You can uncomment the following codes and fill in the blanks
        ### BEGIN SOLUTION
        # YOUR CODE HERE
        self.Qsa[(s,a)] = ( self.Nsa[(s, a)] * self.Qsa[(s,a)] + v ) / (self.Nsa[(s, a)] + 1)
        self.Nsa[(s,a)] += 1
        ### END SOLUTION
        
        # Update the number of times that s has been visited
        self.Ns[s] += 1
        
        return -v

    def getActionProb(self, canonicalBoard):
        
        # Doing self.num_MCTS_sims times of simulations starting from the state 'canonicalBoard'
        for i in range(self.num_MCTS_sims):
            self.search(canonicalBoard)

        # Use string representation for the state
        s = self.game.stringRepresentation(canonicalBoard)
        
        # You can uncomment the following codes and fill in the blanks
        ### BEGIN SOLUTION
        # YOUR CODE HERE
        probs = [0] * self.game.getActionSize()
        
        for a in range(self.game.getActionSize()):
          if (s,a) in self.Nsa:
            probs[a] = self.Nsa[(s,a)] / self.Ns[s]

        
        return probs

     
      



class Agent:

    def __init__(self, game):
        self.game = game
        self.nnet = PolicyNet(game)
        self.pnet = PolicyNet(game)  # the competitor network
        self.mcts = MCTS(game, self.nnet)
        self.epochs = 30  # number of training epochs for each iteration
        self.learning_rate = 0.001
        self.batch_size = 64  # batch size
        self.trainExamples = []  # historical examples for training
        self.numIters = 2  # number of iterations
        self.numEps = 25  # number of complete self-play games for one iteration.
        self.arenaCompare = 40  # number of games to play during arena play to determine if new net will be accepted.
        self.updateThreshold = 0.6  # During arena playoff, new neural net will be accepted if threshold or more of games are won.
        #self.device = torch.device("cuda")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.create_actor()
        
        
    def create_actor(self):
    
        self.nnet.load_state_dict(torch.load('actor_fifth.pth', map_location=torch.device('cpu')))
        
    def play(self, canonicalBoard):
        mcts = MCTS(self.game, self.nnet)
        action = np.argmax(mcts.getActionProb(canonicalBoard))
        return action

    
    
    
def play(canonicalBoard):

    game = OthelloGame(6)
    #nnet = PolicyNet(game)
    #nnet.load_state_dict(torch.load('actor_1.pth', map_location=torch.device('cpu')))
    agent = Agent(game)
    #agent.nnet.load_state_dict(torch.load('actor_first.pth', map_location=torch.device('cpu')))
    
    action = agent.play(canonicalBoard)
    return action




