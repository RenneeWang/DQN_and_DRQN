import random
from collections import deque
import torch
import torch.optim as optim
import numpy as np

from DRQN import *

class Agent:

    def __init__(self, state_size, action_size, bs, lr, tau, gamma, device, visual=False):
        '''
        When dealing with visual inputs, state_size should work as num_of_frame
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.bs = bs
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.device = device
        if visual:
            self.Q_local = VISUAL_DRQN(self.state_size, self.action_size).to(self.device)
            self.Q_target = VISUAL_DRQN(self.state_size, self.action_size).to(self.device)
        else:
            self.Q_local = DRQN(self.state_size, self.action_size).to(device)
            self.Q_target = DRQN(self.state_size, self.action_size).to(device)
        self.soft_update(1)
        self.optimizer = optim.RMSprop(self.Q_local.parameters(), self.lr)
        self.memory = deque(maxlen=100000)
        self.criterion = nn.SmoothL1Loss()
        
        

    def act(self, state, eps=0):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_values = self.Q_local(state)
        if random.random() > eps:
            with torch.no_grad():
                action = q_values.max(-1)[1].detach()[0].item()
            return action # np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
        

    def learn(self):
        experiences = random.sample(self.memory, self.bs)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(self.device)
#         print(states.shape)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(self.device)
        
#         state = torch.tensor(torch.FloatTensor(np.expand_dims(np.expand_dims(states, 0), 0)), dtype=torch.float32).to(self.device)
        states = states.view(-1,states.shape[0],states.shape[1])
        Q_values = self.Q_local(states)
        Q_values = Q_values[-1,:,:]
        Q_values = torch.gather(input=Q_values, dim=-1, index=actions)
        
        next_states = next_states.view(-1,next_states.shape[0],next_states.shape[1])
        with torch.no_grad():
            Q_targets = self.Q_target(next_states)
            Q_targets = Q_targets[-1,:,:]
            Q_targets, _ = torch.max(input=Q_targets, dim=-1, keepdim=True)
            Q_targets = rewards + self.gamma * (1 - dones) * Q_targets

        loss = self.criterion(Q_values,Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        

    def soft_update(self, tau):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)