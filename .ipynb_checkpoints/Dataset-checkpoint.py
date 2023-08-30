from torch.utils.data import Dataset, DataLoader

class NerfDataSet(Dataset):
	def __init__(self, poses, images):
		assert(len(poses) == len(images))

		self.poses = poses
		self.images = images

	def __len__(self) -> int:
		return len(self.poses)

	def __getitem__(self, index):
		return self.poses[index], self.images[index]