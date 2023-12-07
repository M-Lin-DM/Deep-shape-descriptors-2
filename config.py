DATASET_DIR = r"D:\Datasets\Letters deep shape\point clouds"
sigma = 0.24  # std of noise added to each point. 0.08
batch_size = 32
latent_size = 16  # a large latent size will lead to sparse data manifold in latent vector space, but perhaps better denoising performance. Smaller dim should encourage more connected latent space that can be interpolated.
lr = 0.005
lr_model_params = lr * 0.9
lr_latents = lr * 3
epochs = 70
N = 3000
pc_dim = 2  # dimensionality of the point cloud embedding space
log_interval = 10
sigma_z = 0.0  # sigma of noisd added to the latents (optionally)
CHECKPOINT_DIR = r"C:\Users\MrLin\OneDrive\Documents\Experiments\Deep shape descriptor\SAVED MODELS"
model_filename = 'MODEL_letters_augm_2'
latent_filename = 'LATENTS_letters'
bottleneck_dim = 16
alpha = 0.0006   # weight for L2 loss in regularization. VERY SENSITIVE.


hyperparams = dict(epochs=epochs, batch_size=batch_size, alpha=alpha, lr_model_params=lr_model_params, lr_latents=lr_latents, sigma=sigma, latent_size=latent_size, sigma_z=sigma_z, model_filename=model_filename, latent_filename=latent_filename)
