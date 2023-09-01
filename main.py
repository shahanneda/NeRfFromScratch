from NeRF import NeRFManager
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import tensor

data = np.load('tiny_nerf_data.npz')
images = torch.tensor(data['images'])

trainingImages = images[0:100]
valImages = images[100:]

poses = torch.tensor(data['poses'])
trainingPoses = poses[0:100]
valPoses = poses[100:]

focalLength = data['focal']

width = images.shape[1]
height = images.shape[2]

nerf = NeRFManager(trainingImages, trainingPoses, focalLength, numberOfSamples=64, far=6, valImages=valImages, valPoses=valPoses)
nerf.train(epochs=10000)