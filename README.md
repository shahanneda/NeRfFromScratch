# NeRfFromScratch
Implementing NeRF from scratch, by following the [original research paper. ](https://arxiv.org/abs/2003.08934)



I first started out by creating a 2D "nerf", which just memorizes a photo: https://github.com/shahanneda/NeRfFromScratch/blob/main/single-image.ipynb

And then I moved on to making it 3D and an actual nerf with volumetric rendering: https://github.com/shahanneda/NeRfFromScratch/blob/main/3d_image.ipynb 

Then, to run overnight training runs, I extracted some code to classes and more proper python files, and also integrated with weights and biases to run many different training runs with different parameters:
 ![image](https://github.com/shahanneda/NeRfFromScratch/assets/17485954/55f503f2-5c9a-4b01-b32f-95a4fa7e045c)

