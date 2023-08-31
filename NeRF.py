import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import torch.nn as nn
from torch import tensor
from torch.utils.data import Dataset, DataLoader
import gc
import math
from Model import NeRfModel
from Dataset import NerfDataSet
from tqdm import tqdm
from IPython.display import clear_output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NeRFManager():
	def __init__(self, images, poses, focalLength: float, numberOfSamples: int = 25, far: float = 5):
		self.width = images.shape[1]
		self.height = images.shape[2]
		self.focalLength = focalLength
		self.numberOfSamples = 25
		self.far = 5
		self.numberOfFor = 6
		self.model = NeRfModel(self.numberOfFor).to(device)
		self.dataSet = NerfDataSet(poses, images)
		self.dataLoader = DataLoader(dataset=self.dataSet, batch_size=4, shuffle=True )

	def getRays(self, pose):
			xCoords = torch.arange(self.width)
			yCoords = torch.arange(self.height)
			x, y = torch.meshgrid(xCoords, yCoords)

			xShifted = (x - self.width*0.5)/self.focalLength # x coords in a [width, height] tensor

			yShifted = (y - self.height*0.5)/self.focalLength # y coords in a [width, height] tensor

			z = torch.ones_like(x)  # z coords in a [width, height] tensor

			# match up each element of the 3 tensors (thats why using dim = 2)
			directionVectors = torch.stack((xShifted, -yShifted, -z), dim=2).to(device)
			rotationMatrix = pose[0:3, 0:3]
			rotatedDirections = torch.sum(directionVectors.unsqueeze(2)*rotationMatrix, dim=-1)

			origin =  pose[:3, 3]

			# all the rays have the same origin
			originTensor =  torch.broadcast_to(origin, rotatedDirections.shape)
			return rotatedDirections, originTensor


	def get_rays_with_samples(self, pose):
		dirs, pos = self.getRays(pose)
		t = torch.linspace(0, self.far, self.numberOfSamples).reshape(1, 1, self.numberOfSamples, 1).to(device)

		# dirs has shape (width, height, 3) right now (a direction for every pixel)
		# We want to instead have a list of numberOfSamples for each pixel, so (width, height, numberOfSamples, 3)
		dirs = dirs.reshape(self.width, self.height, 1, 3)
		pos = pos.reshape(self.width, self.height, 1, 3)
		z = pos + t*dirs
		z = z.to(device)
		return z

	def get_model_at_each_sample_point(self, rays):
			#rays is (width, height, numberOfSampels, 3), we want to turn the 3 into 15 by appling foruir feature vectors
			raysBackup = torch.clone(rays)
			rays = rays.reshape(self.width, self.height, self.numberOfSamples, 1, 3, 1).expand(self.width, self.height, self.numberOfSamples, 2, 3, 1)
			twos = torch.tensor(2).repeat(self.width, self.height, self.numberOfSamples, 2, 3, self.numberOfFor) # 2 since one for sin one for cos
			twos[:, :, :, :, 0] = 1
			twos = torch.cumprod(twos, dim=4).to(device)

			# Twos is a (3, numberOfFor+, 21) shaped where each row is [1, 2, 4, 8, ...]
			encoding = rays*math.pi*twos
			encoding[:, :, :, 0] = torch.sin(encoding[:, :, :, 0])
			encoding[:, :, :, 1] = torch.cos(encoding[:, :, :, 1])
			encoding = torch.flatten(encoding, start_dim=3, end_dim=5)

			# add non fourer as well (rays backup is just the normal xyz coords)
			encoding = torch.concat((raysBackup, encoding), dim=3)

			return self.model(encoding)
			

	def get_image(self, pose):
		rays = self.get_rays_with_samples(pose)
		return self.get_image_with_rays(rays)

	def get_image_with_rays(self, rays):
		distanceBetweenSamples = self.far / self.numberOfSamples
		out = self.get_model_at_each_sample_point(rays)
		# model_out = (width, height, numberOfSamples, 4), where r is rgb + d
		deltaI = tensor(distanceBetweenSamples)

		# Goes from out = (width,height, numberOfSamples, 4) to C = (width, height, 3)
		# The first two dimenstions are width and height
		Ti = torch.cumprod(torch.exp(-out[:, : , :, 3]*deltaI), dim = 2)
		Ti = Ti.reshape(self.width, self.height, self.numberOfSamples, 1)

		aboservedAmounts = (1- torch.exp(-out[:, :, :, 3]*deltaI)).reshape((self.width, self.height, self.numberOfSamples, 1))
		colorI = out[:, :, :, 0:3]

		# Sum along the ray (the first two dimenensions are the width and height)
		C = torch.sum(Ti*aboservedAmounts * colorI, dim = 2)
		return C
	
	def train(self, epochs, filePath="./model.pkl"):
		def loss_fn(output, target):
				loss = torch.mean((output - target)**2)
				return loss

		optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
		pbar = tqdm(range(epochs))

		best_loss = float("inf")
		loss = 0
		for epoch in pbar: 
				epoch_loss = 0
				torch.cuda.empty_cache()

				for (pose_batch, image_batch) in self.dataLoader:
						pose_batch = pose_batch.to(device)
						image_batch = image_batch.to(device)
						for (pose, image) in zip(pose_batch, image_batch):
								rays = self.get_rays_with_samples(pose)
								pred_image = self.get_image_with_rays(rays)

								image = image.to(device)
								loss += loss_fn(pred_image, image)
								epoch_loss += loss.item()

						optimizer.zero_grad()
						loss.backward()
						optimizer.step()
						loss = 0

				pbar.set_description(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}")
				if(epoch_loss < best_loss):
					torch.save(self.model.state_dict(), filePath)
					best_loss = epoch_loss