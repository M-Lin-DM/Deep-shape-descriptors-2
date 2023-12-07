import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from dataloader import *
from deeplatent import *
from networks import *
from utils import *
from config import *

device = "cuda" if torch.cuda.is_available() else "cpu"
# load_file = latent_filename
load_file = model_filename

dataset = Letters(DATASET_DIR, device, sigma)  # returns an entire point cloud [N, 3] as the training instance
loader = DataLoader(dataset, batch_size=8, shuffle=False)
shape_batch, shape_gt_batch, latent_indices = next(iter(loader))  # the noise added is random each time, so there's a kind of automatic data augmentation going on
print(f"shape of model input: {shape_batch.shape}")

# randomize point cloud to uniform distribution in 2D
# shape_batch = shape_batch + torch.randn_like(shape_batch) * 0.0  #adding noise somehow changes the models prediction meaning that points are not independent of each other
# shape_batch = torch.rand_like(shape_batch) * 2 - 1
# shape_batch = shape_gt_batch


num_total_instance = len(dataset)
num_batch = len(loader)
print(f"num_batch {num_batch}")

model = DeepLatent(latent_length=latent_size, n_points_per_cloud=N, chamfer_weight=0.1)
model, latent_vecs, optimizer = load_checkpoint(os.path.join(CHECKPOINT_DIR, load_file), model, None)

# try noiseing latent vects to see if it affects the predictions
# for i, v in enumerate(latent_vecs):
#     latent_vecs[i] = torch.rand_like(latent_vecs[i]) * 2 - 1

# print(latent_vecs)
shape_batch.to(device)
shape_gt_batch.to(device)
model.to(device)

latent_repeat = contruct_latent_repeat_tensor(shape_batch, latent_indices, latent_vecs, device='cuda', use_noise=False)
# print(latent_repeat)

# latent_vecs is a list of tenors of length 128
# print(shape_batch.shape, latent_repeat)

pc_list = pc_batch_to_data_matrices_list(shape_batch)
pc_gt_list = pc_batch_to_data_matrices_list(shape_gt_batch)
#
loss, pc_denoised = model(shape_batch, shape_gt_batch, latent_repeat)
pc_denoised_list = pc_batch_to_data_matrices_list(pc_denoised)

q = 2
# plt.scatter(pc_gt_list[q][:, 0], pc_gt_list[q][:, 1], color='red', s=4)
plt.scatter(pc_list[q][:, 0], pc_list[q][:, 1], color='blue', s=2)
plt.scatter(pc_denoised_list[q][:, 0], pc_denoised_list[q][:, 1], color='green', s=4)

# Set the labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('')
plt.axis('equal')
# Show the plot
plt.show()

# print(pc_batch_to_data_matrices_list(pc_denoised)[0].shape)

