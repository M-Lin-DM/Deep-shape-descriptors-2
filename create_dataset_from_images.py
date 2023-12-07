from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from utils import *
import random

DATASET_DIR = r"D:\Datasets\Letters deep shape\point clouds"
train_img_dir = r"D:\Datasets\Letters deep shape\imgs\letter"
train_img_path = Path(train_img_dir)
img_files = list(train_img_path.glob('*.png'))  # will list the entire path
N_reps = 1000  # number of copies of each letter present in the dataset. This is the data augmentation factor. 1.2GB for 26 * 1000 point dataset

N = 3000  # number of points to sample from the list of extracted pixel coords
pc_list = []

for j, image_path in enumerate(img_files):
    print(j, image_path)
    if j < 5000:
        image = Image.open(image_path).convert('1')  # convert to binary
        img_arr = np.array(image)

        # print(np.sum(~img_arr))
        # Find the coordinates of all pixels with the value=0
        row_indices, col_indices = np.where(img_arr == 0)
        if len(row_indices) < N:  # removes small shapes and ensures we have a uniform array
            continue

        # Combine row and column indices into a single array of coordinates
        coordinates = np.vstack((col_indices, -row_indices)).T  # NOTE: this treats the row (ie y) as the x coord so the image will appear rotated 90 from the initial png image
        coordinates = center_and_rescale(coordinates)

        # For each image, apply random rotation in specified range and sample a random subset of points
        for rep in range(N_reps):
            coordinates_rotated = rotate_points(coordinates)
            # sample fixed number of points N
            sample_inds = np.random.choice(len(coordinates), size=(N,), replace=False)
            pc_list.append(coordinates_rotated[sample_inds])

        print(len(coordinates[sample_inds]))
        # # plt.imshow(img_arr)
        # plt.scatter(coordinates[sample_inds, 0], coordinates[sample_inds, 1], color='black', s=4)
        # # plt.scatter(dat_gt[:, 0], dat_gt[:, 1], color='red', s=4)
        #
        # # Set the labels and title
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Scatter Plot')
        # plt.axis('equal')
        # # Show the plot
        # plt.show()

print(f"dataset length: {len(pc_list)}")
random.shuffle(pc_list)  # shuffle elements. OR pc_list[np.random.permutation(len(pc_list))]
np.save(f'{DATASET_DIR}\\pc_list.npy', pc_list, allow_pickle=True)  # converts list to numpy array