from dataloader import Flowers
from config import *
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


dataset = Flowers(DATASET_DIR, "cuda", sigma=sigma)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

shape_batch, shape_gt_batch, latent_indices = next(iter(dataset))

print(len(loader))
dat = shape_batch.cpu().numpy().transpose()
dat_gt = shape_gt_batch.cpu().numpy().transpose()

plt.scatter(dat[:, 0], dat[:, 1], color='black', s=4)
plt.scatter(dat_gt[:, 0], dat_gt[:, 1], color='red', s=4)

# Set the labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot')
plt.axis('equal')
# Show the plot
plt.show()