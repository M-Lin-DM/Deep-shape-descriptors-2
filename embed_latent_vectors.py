import numpy as np
import matplotlib.pyplot as plt

from dataloader import *
from deeplatent import *
from utils import *
from config import *

# load_file = latent_filename
load_file = model_filename

device = "cuda" if torch.cuda.is_available() else "cpu"

save_name = os.path.join(CHECKPOINT_DIR, load_file)
z_ts = torch.load(save_name + '_latents.pt').cpu().detach().numpy()
print(z_ts.shape)
# print(z_ts)

# emb = embed_tsne(z_ts, initial_pos=None)
emb = embed_umap(z_ts)

kmeans, labels, cluster_centers = kmeans_clustering(z_ts, 26)
# scatter3D(emb[:, :3], labels)
# scatter3D(z_ts[:, :3], labels)
# sys.exit()  # terminate execution

dataset = Letters(DATASET_DIR, device, sigma, rotation=False)  # returns an entire point cloud [N, 3] as the training instance
# shape_batch, shape_gt_batch, latent_indices = next(iter(dataset))

num_clusters = 16
grid_size = 8
num_images = int(grid_size ** 2)  # number of images to plot
fig_scale = 5
fig_width = grid_size * fig_scale
fig_height = grid_size * fig_scale

# plot shapes from each cluster
for c in range(num_clusters):
    cluster_ind = np.where(labels == c)[0]  # numpy array of indices of data points sharing a certain cluster ID
    # cluster_ind = np.random.choice(26000, size=num_images)
    # print(len(cluster_ind))

    pc, pc_gt, index = dataset[
        cluster_ind]  # NOTE: for some reason, this transposes the last 2 dims. shapes= List[(N,2)]
    pc_gt = pc_gt.cpu().detach().numpy()  # shape=(num points, cluster size, 2) Not sure why..
    pc_gt = pc_gt.transpose(1, 0, 2)  # shape=(batch, #points in cloud, 2)
    print(pc_gt.shape)

    plt.figure(figsize=(fig_width, fig_height))
    plt.axis('off')

    for j, dat in enumerate(pc_gt[:num_images]):
        # print(dat.shape)
        plt.subplot(grid_size, grid_size, j + 1, aspect='equal')
        plt.scatter(dat[:, 0], dat[:, 1], color='black', s=6)
        plt.axis('off')
    #
    # plt.show()
    plt.savefig(f"C:\\Users\\MrLin\\OneDrive\\Documents\\Experiments\\Deep shape descriptor\\fig\\cluster_{c}.png",
                bbox_inches='tight', pad_inches=0, dpi=120)
    plt.close()
