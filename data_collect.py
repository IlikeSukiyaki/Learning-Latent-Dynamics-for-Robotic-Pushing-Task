# Collect data (it may take some time)
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from IPython.display import Image
from tqdm import tqdm
from learning_state_dynamics import collect_data_random

from panda_pushing_env import PandaPushingEnv
# Data collection parameters
N = 100 # Number of trajectories
T = 10 # Trajectory length

# Initialize the environment and collect data
env = PandaPushingEnv()
env.reset()
collected_data = collect_data_random(env, num_trajectories=N, trajectory_length=T)


# Verify the number of data collected:
print(f'We have collected {len(collected_data)} trajectories')
print('A data sample contains: ')
for k, v in collected_data[0].items():
    assert(type(v) == np.ndarray)
    assert(v.dtype == np.float32)
    print(f'\t {k}: numpy array of shape {v.shape}')

# Save the collected data into a file
np.save(os.path.join("/home/yifeng/PycharmProjects/Learning_Dynamics/learning_dynamics_1", 'collected_data.npy'), collected_data)
