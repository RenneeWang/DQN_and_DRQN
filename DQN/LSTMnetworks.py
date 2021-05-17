import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_Q_Network(nn.Module):

    def __init__(self, state_size, action_size, hidden=[64, 64]):
        super(LSTM_Q_Network, self).__init__()
        self.batchnorm1 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True)
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.lstm = nn.LSTM(input_size=hidden[0], hidden_size=hidden[1], num_layers=20, batch_first=True)
        self.fc3 = nn.Linear(hidden[1], action_size)

    def forward(self, state): 
        x = state
        x = torch.relu(self.fc1(x))
        x = x.view(-1,128,64) if len(x.shape) == 2 else x.view(-1,64,1)
        out, self.hidden_cell = self.lstm(x)
        out = self.batchnorm1(out)
        out = out[-1,:,:]
        x = F.relu(out)
        x = self.fc3(x)
        return x

class Q_Network(nn.Module):

    def __init__(self, state_size, action_size, hidden=[64, 64]):
        super(LSTM_Q_Network, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size)

    def forward(self, state):
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Visual_Q_Network(nn.Module):
    '''
    The input of this network should have shape (num_frame, 80, 80)
    '''

    def __init__(self, num_frame, num_action):
        super(Visual_Q_Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_frame, out_channels=16, kernel_size=8, stride=4, padding=2)  # 16, 20, 20
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)  # 32, 9, 9
        self.fc1 = nn.Linear(32 * 81, 256)
        self.fc2 = nn.Linear(256, num_action)

    def forward(self, image):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 81)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x