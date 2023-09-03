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
import PIL
import torchvision.transforms.functional
import wandb


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class NeRFManager():
	def __init__(self, images, poses, focalLength: float, numberOfSamples: int, near: float = 2, far: float = 6, valImages = None, valPoses = None):
		assert(len(images) == len(poses))
		assert(len(valImages) == len(valPoses))

		self.width = images.shape[1]
		self.height = images.shape[2]
		self.focalLength = focalLength
		self.numberOfSamples = numberOfSamples
		self.near = near
		self.far = far
		self.numberOfFor = 6
		self.layerSize = 256
		self.numberOfLayers = 8
		self.model = NeRfModel(self.numberOfFor, layer_size=self.layerSize, number_of_layers=self.numberOfLayers).to(device)
		self.dataSet = NerfDataSet(poses, images)
		self.batch_size = 1
		self.dataLoader = DataLoader(dataset=self.dataSet, batch_size=self.batch_size, shuffle=True )
		self.valImages = valImages
		self.valPoses = valPoses
		self.lr = 5e-4

		self.setup_weights_and_bias()



	def setup_weights_and_bias(self):
		wandb.init(
			# set the wandb project where this run will be logged
			project="NerfFromScratch",
			
			# track hyperparameters and run metadata
			config={
			"number_of_fourier": self.numberOfFor,
			"number_of_samples": self.numberOfSamples,
			"layer_size": self.layerSize,
			"number_of_layers": self.numberOfLayers,
			"near": self.near,
			"far": self.far,
			"learning_rate": self.lr,
			"batch_size": self.batch_size,
			}
		)

	def get_rays(self, pose):
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


	def get_sample_points_and_distances(self, pose):
		dirs, pos = self.get_rays(pose)
		t = torch.linspace(self.near, self.far, self.numberOfSamples).reshape(1, 1, self.numberOfSamples, 1).to(device)
		stepSize = (self.far-self.near)/self.numberOfSamples
		noise = torch.rand_like(t)*stepSize
		t = t + noise


		dists = t.squeeze(3)[:, :,  1:] - t.squeeze(3)[:, :, : -1]
		infiniteLastRay = torch.tensor(1e10).broadcast_to(1, 1, 1).to(device)
		dists = torch.concat([dists, infiniteLastRay], dim=2)
		dists = dists.broadcast_to((self.width, self.height, self.numberOfSamples))

		# dirs has shape (width, height, 3) right now (a direction for every pixel)
		# We want to instead have a list of numberOfSamples for each pixel, so (width, height, numberOfSamples, 3)
		dirs = dirs.reshape(self.width, self.height, 1, 3)
		pos = pos.reshape(self.width, self.height, 1, 3)
		z = pos + t*dirs
		z = z.to(device)
		return (z, dists)

	def get_model_at_each_sample_point(self, samplePoints):
			#rays is (width, height, numberOfSampels, 3), we want to turn the 3 into 15 by appling foruir feature vectors
			originalSamplePoints = torch.clone(samplePoints)
			samplePoints = samplePoints.reshape(self.width, self.height, self.numberOfSamples, 1, 3, 1).expand(self.width, self.height, self.numberOfSamples, 2, 3, 1)
			twos = torch.tensor(2).repeat(self.width, self.height, self.numberOfSamples, 2, 3, self.numberOfFor) # 2 since one for sin one for cos
			twos[:, :, :, :, 0] = 1
			twos = torch.cumprod(twos, dim=4).to(device)

			# Twos is a (3, numberOfFor+, 21) shaped where each row is [1, 2, 4, 8, ...]
			encoding = samplePoints*twos
			encoding[:, :, :, 0] = torch.sin(encoding[:, :, :, 0])
			encoding[:, :, :, 1] = torch.cos(encoding[:, :, :, 1])
			encoding = torch.flatten(encoding, start_dim=3, end_dim=5)

			# add non fourer as well (rays backup is just the normal xyz coords)
			encoding = torch.concat((originalSamplePoints, encoding), dim=3)

			return self.model(encoding)
			

	def get_image(self, pose):
		samplePoints, dists = self.get_sample_points_and_distances(pose)
		return self.get_image_with_sample_points_and_distances(samplePoints, dists)

	def get_image_with_sample_points_and_distances(self, samplePoints, dists):
		out = self.get_model_at_each_sample_point(samplePoints)

		densityFromModel = out[:, :, :, 3]
		rgbFromModel = out[:, :, :, 0:3]

		expOfAlphaAndDists = torch.exp(-densityFromModel*dists)

		Ti = torch.cumprod(expOfAlphaAndDists, dim = 2)
		Ti = Ti.reshape(self.width, self.height, self.numberOfSamples, 1)

		absorbedAmounts = (1 - expOfAlphaAndDists).reshape((self.width, self.height, self.numberOfSamples, 1))
		# Sum along the ray (the first two dimenensions are the width and height)
		C = torch.sum(Ti*absorbedAmounts * rgbFromModel, dim = 2)
		C = C.rot90(k=-1)
		return C
	
	def get_wb_image(self, img):
					return wandb.Image(
							torchvision.transforms.functional.to_pil_image(img.cpu().permute(2, 1, 0)),
					)
	
	def train(self, epochs, filePath="./model.pkl", resumeFromFile=None):
		if(resumeFromFile):
			print("Resuming from: ", resumeFromFile)
			self.model.load_state_dict(torch.load(resumeFromFile))

		def loss_fn(output, target):
				loss = torch.mean((output - target)**2)
				return loss

		optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
		pbar = tqdm(range(epochs))

		best_loss = float("inf")
		for epoch in pbar: 
				loss = 0
				epoch_loss = 0
				torch.cuda.empty_cache()

				for (pose_batch, image_batch) in self.dataLoader:
						pose_batch = pose_batch.to(device)
						image_batch = image_batch.to(device)
						for (pose, image) in zip(pose_batch, image_batch):
								pred_image = self.get_image(pose)
								image = image.to(device)
								loss += loss_fn(pred_image, image)
								epoch_loss += loss.item()

						optimizer.zero_grad()
						loss.backward()
						optimizer.step()
						loss = 0
				
				val_loss = 0
				with torch.no_grad():
					training_img = self.get_wb_image(pred_image)
					validation_images = []
					for (pose, image) in zip(self.valPoses, self.valImages):
									pose = pose.to(device)
									image = image.to(device)
									pred_image = self.get_image(pose)
									image = image.to(device)
									val_loss += loss_fn(pred_image, image)
									validation_images.append(pred_image.cpu())


					wandb.log({"epoch": epoch, "epoch_loss": epoch_loss, "val_loss": val_loss})
					wandb.log({"validation_images": [ self.get_wb_image(img) for img in validation_images], "training_image": training_img})
					pbar.set_description(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}, Val Loss: {val_loss}")

				if(val_loss < best_loss):
					torch.save(self.model.state_dict(), filePath)
					best_loss = val_loss