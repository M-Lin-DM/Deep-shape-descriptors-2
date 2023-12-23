import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import wandb

from dataloader import *
from deeplatent import *
from utils import *
from config import *

wandb.login()
wandb.init(project="deep shape Letters",
           config=hyperparams)  # this initializes a new run on wandb. config=hyperparams sets wandb.config, a dictionary-like object. hyperparams is a dictionary.
wconfig = wandb.config

resume = False  # resume training
load_file = model_filename
save_file = model_filename

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = Letters(DATASET_DIR, device, wconfig.sigma, rotation=True)  # returns an entire point cloud [N, 3] as the training instance
# dataset_val = Flowers_val(DATASET_DIR, device, wconfig.sigma)
loader = DataLoader(dataset, batch_size=wconfig.batch_size, shuffle=True)
# loader_val = DataLoader(dataset_val, batch_size=wconfig.batch_size, shuffle=False)

shape_batch, shape_gt_batch, latent_indices = next(iter(loader))
print(f"shape of model input: {shape_batch.shape}")
assert shape_batch.shape[2] == N
num_total_instance = len(dataset)
num_batch = len(loader)
print(f"num_batch {num_batch}")

model = DeepLatent(latent_length=wconfig.latent_size, n_points_per_cloud=N, chamfer_weight=0.1)
model.to(device)  # this need to be placed before* loading model checkpoint, otherwise tensors will not be on the same device.
latent_vecs = initialize_latent_vecs(dataset, device)  # random initialization. These will be overwritten if loading checkpoint

optimizer = optim.Adam([
        {
            "params": model.parameters(), "lr": wconfig.lr_model_params,
        },
        {
            "params": latent_vecs, "lr": wconfig.lr_latents,
        }
    ]
    )

if resume:
    model, latent_vecs, optimizer = load_checkpoint(os.path.join(CHECKPOINT_DIR, load_file), model, optimizer, device)  # this should rewrite latent_vecs, loads state dict into the model and optimizer

else:
    model.apply(init_weights)  # initialize weights. This can affect ability to find optima
    # wandb.watch(model)  # log gradients

print("seed", torch.random.seed())
# min_loss = float('inf')

for epoch in range(wconfig.epochs):
    print(f"epoch {epoch}")
    training_loss = 0.0
    validation_loss = 0.0

    model.train()
    for index, (shape_batch, shape_gt_batch, latent_indices) in enumerate(
            loader):  # in my code I need a step to select a fixed number of sample points if the pcs have different sizes.
        # latent_indices is a batch of indices in the dataset (batch_size,). each index corresponds to one entire point cloud
        # shape_batch: shape=(batch_size, 3, N)
        shape_batch.requires_grad = False
        shape_gt_batch.requires_grad = False

        latent_repeat = contruct_latent_repeat_tensor(shape_batch, latent_indices, latent_vecs, device='cuda',
                                                      use_noise=False, wconfig=wconfig)

        shape_batch = shape_batch.to(device)
        shape_gt_batch = shape_gt_batch.to(device)
        (loss, chamfer, l2), pc_est = model(shape_batch, shape_gt_batch, latent_repeat)

        # Compute l2 loss component. This is different from the l2 used as a distance between the point clouds
        # the behavioral mode during training is very sensitive to this parameter. 0/Low values cause model to ignore latent vects. high values reduce model performance
        l_parameters = []
        for param in model.parameters():
            l_parameters.append(param.view(-1))  # -1 flattens the param tensor

        # apply l2 loss to specific layers!
        # for name, param in model.named_parameters():
        #     if name in ['pdl_net.conv1.weight', 'pdl_net.conv1.bias', 'pdl_net.conv4.weight', 'pdl_net.conv4.bias']:
        #         l_parameters.append(param.view(-1))  # -1 flattens the param tensor

        L2 = wconfig.alpha * model.compute_L2_regularization_loss(
            torch.cat(l_parameters))  # cat all params into a single vector
        # Add l2 loss component
        loss += L2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

        print("Epoch:[%d|%d], Batch:[%d|%d]  loss: %f , chamfer: %f, l2: %f" % (
            epoch, wconfig.epochs, index, num_batch, loss.item(), chamfer.item(), l2.item()))

    # get validation loss
    # for index, (shape_batch, shape_gt_batch, latent_indices) in enumerate(loader_val):
    #     latent_repeat = contruct_latent_repeat_tensor(shape_batch, latent_indices, latent_vecs, device='cuda',
    #                                                   use_noise=False, wconfig=wconfig)
    #     (loss, chamfer, l2), pc_est = model(shape_batch, shape_gt_batch, latent_repeat)
    #     validation_loss += loss.item()

    training_loss_epoch = training_loss / len(loader)  # loss per batch (ie set of point cloud)
    # validation_loss_epoch = validation_loss / len(loader_val)

    z_tensor = torch.stack(latent_vecs, dim=1).transpose(0,
                                                         1)  # stack concats along a NEW dimension. torch.cat concats along EXISTING dim
    # print(z_tensor.shape)
    z_cloud_std = torch.norm(torch.std(z_tensor, dim=0), p=2)
    conv1 = model.state_dict()['pdl_net.conv1.weight'].squeeze().cpu().detach().numpy()[
            :20]  # weights from first conv layer
    z_tensor_samp = z_tensor[:20].cpu().detach().numpy()

    # wandb.log({"train": {"acc": 0.9}, "val": {"acc": 0.8}})
    wandb.log({"train-loss": training_loss_epoch, "z_cloud_std": z_cloud_std, "conv1_weights": wandb.Image(conv1),
               "latent_vecs": wandb.Image(z_tensor_samp), "L2 reg": L2})

    # if training_loss_epoch < min_loss:
    #     min_loss = training_loss_epoch
    #     print('New best performance! saving')
    #     save_name = os.path.join(CHECKPOINT_DIR, 'model_best')
    #     save_checkpoint(save_name, model, latent_vecs, optimizer)
    #
    # if (epoch + 1) % log_interval == 0:
    #     save_name = os.path.join(CHECKPOINT_DIR, 'model_routine')
    #     save_checkpoint(save_name, model, latent_vecs, optimizer)

save_name = os.path.join(CHECKPOINT_DIR, save_file)
save_checkpoint(save_name, model, latent_vecs, optimizer)
