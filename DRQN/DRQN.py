import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import gym
import ConvLSTM


class DRQN(nn.Module):
    def __init__(self, observation_dim, action_dim, time_step=1, layer_num=1, hidden_num=128):
        super(DRQN, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.time_step = time_step
        self.layer_num = layer_num
        self.hidden_num = hidden_num
        self.lstm = nn.LSTM(self.observation_dim, self.hidden_num, self.layer_num, batch_first=True).cuda()
        self.fc1 = nn.Linear(self.hidden_num, 128)
        self.fc2 = nn.Linear(128, self.action_dim)

    def forward(self, state):
#         if not hidden:
#             h0 = torch.zeros([self.layer_num, state.size(0), self.hidden_num])
#             c0 = torch.zeros([self.layer_num, state.size(0), self.hidden_num])
#             hidden = (h0, c0)

        x, new_hidden = self.lstm(state)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    
    

class VISUAL_DRQN(nn.Module):
    '''
    figure
    '''
    def __init__(self, observation_dim, action_dim, time_step=1, layer_num=1, hidden_num=128):
        super(VISUAL_DRQN, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.time_step = time_step
        self.layer_num = layer_num
        self.hidden_num = hidden_num
        self.ConvLSTM0 = nn.ConvLSTM(self.observation_dim, self.hidden_num, self.layer_num, batch_first=True)
        self.fc1 = nn.Linear(self.hidden_num, 128)
        self.fc2 = nn.Linear(128, self.action_dim)

    def forward(self, state):
        
        x, new_hidden = self.ConvLSTM0(state)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# class recurrent_replay_buffer(object):
#     def __init__(self, capacity, seq_len=50):
#         self.capacity = capacity
#         self.seq_len = seq_len
#         self.memory = deque(maxlen=self.capacity)
#         self.memory.append([])

#     def store(self, observation, action, reward, next_observation, done):
#         observation = np.expand_dims(observation, 0)
#         next_observation = np.expand_dims(next_observation, 0)

#         self.memory[-1].append([observation, action, reward, next_observation, done])

#         if len(self.memory[-1]) == self.seq_len:
#             self.memory.append([])

#     def sample(self, batch_size=32):
#         observation = []
#         action = []
#         reward = []
#         next_observation = []
#         done = []
#         batch = random.sample(list(self.memory)[: -1], batch_size)
#         for i in range(batch_size):
#             obs, act, rew, next_obs, don = zip(* batch[i])
#             obs = np.expand_dims(np.concatenate(obs, 0), 0)
#             next_obs = np.expand_dims(np.concatenate(next_obs, 0), 0)
#             observation.append(obs)
#             action.append(act)
#             reward.append(rew)
#             next_observation.append(next_obs)
#             done.append(don)
#         return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done

#     def __len__(self):
#         return len(self.memory)



