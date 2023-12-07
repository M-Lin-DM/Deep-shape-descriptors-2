import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import wandb
import time

from dataloader import *
from deeplatent import *
from networks import *
from utils import *
from config import *


load_file = model_filename
save_file = latent_filename

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = Flowers(DATASET_DIR, device, sigma)  # returns an entire point cloud [N, 3] as the training instance
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
shape_batch, shape_gt_batch, latent_indices = next(iter(loader))
# print(f"shape of model input: {shape_batch.shape}")
assert shape_batch.shape[2] == N
num_total_instance = len(dataset)
num_batch = len(loader)
# print(f"num_batch {num_batch}")

model = DeepLatent(latent_length=latent_size, n_points_per_cloud=N, chamfer_weight=0.1)

# initialize all latent vectors in the dataset
latent_vecs = initialize_latent_vecs(dataset, device)

# print("before training:", latent_vecs[0])
# time.sleep(5)

optimizer = optim.Adam([
    {
        "params": latent_vecs, "lr": lr * 3,
    }
]
)

model, _, _ = load_checkpoint(os.path.join(CHECKPOINT_DIR, load_file), model, optimizer)

print(model.state_dict().keys())
print(model.state_dict()['pdl_net.conv1.weight'].shape)
conv1 = model.state_dict()['pdl_net.conv1.weight'].squeeze().cpu().detach().numpy()

plt.imshow(conv1, cmap='inferno')
plt.colorbar()
plt.show()

for name, param in model.named_parameters():
    if name in ['pdl_net.conv1.weight', 'pdl_net.conv1.bias', 'pdl_net.conv4.weight', 'pdl_net.conv4.bias']:
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:5]} \n")

# b = [p.shape for p in model.parameters()]
# print(b)
