import torch

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

# soft_update_freq = 100
# capacity = 10000
# exploration = 300
# render = False
# seq_len = 50

#Agent parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TAU = 0.001
GAMMA = 0.99

#environment names
# RAM_ENV = 'Skiing-ramDeterministic-v4'
RAM_ENV = 'LunarLander-v2'
VISUAL_ENV = 'SkiingDeterministic-v4'
CONSTANT = 90


#Training parameters
RAM_NUM_EPISODE = 1000
VISUAL_NUM_EPISODE = 3000
EPS_INIT = 0.9
EPS_DECAY = 0.99
EPS_MIN = 0.05
MAX_T = 1500
NUM_FRAME = 2