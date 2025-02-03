import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from utils.helper_functions import load_config
from src.model import PatchGANDiscriminator
from torch.utils.data import DataLoader
from data.dataset import CustomDataset
from torchvision import transforms

def analyze_discriminator(config, input_dir, output_file):

    # load discriminator
    discriminator = PatchGANDiscriminator()
    
    # Use the same paths as in train_apply.py
    best_discriminator_path = os.path.join(config['logging']['checkpoint_dir'], 'best_model/discriminator_latest.pth')
    
    device = torch.device("cuda" if config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")

    # Load with appropriate device mapping
    discriminator.load_state_dict(
        torch.load(best_discriminator_path, device, weights_only=True)
    )
    discriminator = discriminator.to(device)
    discriminator.eval()

    # Prepare the test dataset and dataloader
    real_dataset = CustomDataset(
        images_dir=f"{input_dir}/1",
        labels_dir=f"{input_dir}/0",
        transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
        is_training=False
    )
    real_loader = DataLoader(real_dataset, batch_size=len(real_dataset), shuffle=False)

    fake_dataset = CustomDataset(
        images_dir=f"{input_dir}/2",
        labels_dir=f"{input_dir}/0",
        transform=transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()]),
        is_training=False
    )
    fake_loader = DataLoader(fake_dataset, batch_size=len(fake_dataset), shuffle=False)

    # Run predictions and store results
    with torch.no_grad():
        for data in real_loader:
            real_A = data['A'].to(device)  # Input image
            real_B = data['B'].to(device)  # Ground truth image

            real_features = discriminator(real_B, real_A, return_features=True)

    with torch.no_grad():
        for data in fake_loader:
            real_A = data['A'].to(device)  # Input image
            fake_B = data['B'].to(device)  # fake image

            fake_features = discriminator(fake_B, real_A, return_features=True)

    # Flatten the feature maps for both real and fake images
    real_features_flat = real_features.view(real_features.size(0), -1).detach().cpu().numpy()
    fake_features_flat = fake_features.view(fake_features.size(0), -1).detach().cpu().numpy()

    # Concatenate the real and fake feature maps into one array
    all_features = np.concatenate([real_features_flat, fake_features_flat], axis=0)

    # Create labels for visualization (1 for real, 0 for fake)
    labels = np.concatenate([np.ones(real_features_flat.shape[0]), np.zeros(fake_features_flat.shape[0])])

    # Apply t-SNE
    tsne = TSNE(n_components=2)  # Reduce to 2D
    tsne_results = tsne.fit_transform(all_features)

    # Split t-SNE results based on labels
    real_tsne = tsne_results[labels == 1]
    fake_tsne = tsne_results[labels == 0]

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(real_tsne[:, 0], real_tsne[:, 1], c='blue', label='Real', marker='o')
    plt.scatter(fake_tsne[:, 0], fake_tsne[:, 1], c='red', label='Fake', marker='x')

    plt.title("t-SNE Visualization: Real & Fake Images")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.savefig(output_file)


if __name__ == "__main__":

    # Load configuration
    config_path = "config.yaml"
    config = load_config(config_path)

    # Define input directory
    input_dir = "results/prototyp1"

    # Define output folder and file
    output_file = "results/prototyp1/tsne.pdf"

    analyze_discriminator(config, input_dir, output_file)
