import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import torch.nn as nn
from torch import tensor

class NeRfModel(nn.Module):
	def __init__(self, num_fourier_features, layer_size, number_of_layers):
		super().__init__()
		self.layer1 = nn.Linear(3 + 6*num_fourier_features, layer_size)

		self.layers = nn.ModuleList([nn.Linear(layer_size, layer_size) for _ in range(number_of_layers)])

		self.rgb_layer = nn.Linear(layer_size, 3)
		self.radiance_layer = nn.Linear(layer_size, 1)

	def forward(self, x):
		x = nn.functional.relu(self.layer1(x))

		for layer in self.layers:
			x = nn.functional.relu(layer(x))

		rgb = torch.sigmoid(self.rgb_layer(x))
		radiance = torch.relu(self.radiance_layer(x))
		return torch.cat([rgb, radiance], dim=-1)