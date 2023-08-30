from NeRF import NeRFManager
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import tensor

data = np.load('tiny_nerf_data.npz')
images = torch.tensor(data['images'])
poses = torch.tensor(data['poses'])
focalLength = data['focal']

poses[12]
print(images.shape)
print(poses.shape)

width = images.shape[1]
height = images.shape[2]

nerf = NeRFManager(images, poses, focalLength, numberOfSamples=25, far=5)
nerf.train(epochs=10000)