import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import torch.nn as nn
from torch import tensor

class NeRfModel(nn.Module):
	def __init__(self, num_fourier_features, layer_size=400):
		super().__init__()
		self.layer1 = nn.Linear(3 + 6*num_fourier_features, layer_size)
		self.layer2 = nn.Linear(layer_size, layer_size)
		self.layer3 = nn.Linear(layer_size, layer_size)
		self.layer4 = nn.Linear(layer_size, layer_size)
		self.layer4 = nn.Linear(layer_size, layer_size)
		self.rgb_layer = nn.Linear(layer_size, 3)
		self.radiance_layer = nn.Linear(layer_size, 1)

	def forward(self, x):
		x = nn.functional.relu(self.layer1(x))
		x = nn.functional.relu(self.layer2(x))
		x = nn.functional.relu(self.layer3(x))
		x = nn.functional.relu(self.layer4(x))

		rgb = torch.sigmoid(self.rgb_layer(x))
		radiance = self.radiance_layer(x)
		return torch.cat([rgb, radiance], dim=-1)