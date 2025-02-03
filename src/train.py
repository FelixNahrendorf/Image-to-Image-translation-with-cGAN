import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.helper_functions import load_config
from data.dataset import CustomDataset
from src.model import UNetGenerator, PatchGANDiscriminator, initialize_weights
from utils.losses import GANLosses
from utils.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.metrics import structural_similarity as calculate_ssim



def train_model(config):
    """
    Train the generator and discriminator using the specified configuration.
    """
    # Add at the start of your train_model function in train.py
    torch.autograd.set_detect_anomaly(True)

    # Prepare datasets
    train_dataset = CustomDataset(
    images_dir=config['data']['train_images_dir'],
    labels_dir=config['data']['train_labels_dir'],
    transform=None,
    is_training=True
    )

    val_dataset = CustomDataset(
        images_dir=config['data']['val_images_dir'],
        labels_dir=config['data']['val_labels_dir'],
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
        is_training=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config['current_training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['current_training']['batch_size'], shuffle=False)

    # Initialize generator and discriminator
    generator = UNetGenerator(dropout_rate=0.5)
    generator.apply(initialize_weights)
    discriminator = PatchGANDiscriminator()
    discriminator.apply(initialize_weights)

    # Set up device for training
    device = torch.device("cuda" if config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")
    generator.to(device)
    discriminator.to(device)

    # Initialize optimizers and schedulers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=config['current_training']['lr'], betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=config['current_training']['lr'], betas=(0.5, 0.999))

    # Get scheduler values from config
    s_factor = config['training']['scheduler'].get('factor', 0.5)
    s_patience = config['training']['scheduler'].get('factor', 5)
    s_min_lr = config['training']['scheduler'].get('min_lr', 0.00001)

    # Initialize learning rate schedulers
    scheduler_G = ReduceLROnPlateau(optimizer_G, mode='min', factor=s_factor, patience=s_patience, verbose=True, min_lr=s_min_lr)
    scheduler_D = ReduceLROnPlateau(optimizer_D, mode='min', factor=s_factor, patience=s_patience, verbose=True, min_lr=s_min_lr)

    # Initialize loss functions
    losses = GANLosses(lambda_L1=config['current_training']['lambda_L1'], gan_mode='vanilla')

    # Get max_grad_norm from config
    max_grad_norm = config['training'].get('max_grad_norm', 1.0)

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=config['training'].get('early_stopping_patience', 7),
        min_delta=config['training'].get('early_stopping_min_delta', 0.0),
        verbose=True
    )

    # Current model as string
    current_model = f"lr{config['current_training']['lr']}_bs{config['current_training']['batch_size']}_lambda{config['current_training']['lambda_L1']}"
    os.makedirs(f"{config['logging']['checkpoint_dir']}/{current_model}", exist_ok=True)

    # Training loop
    for epoch in range(config['training']['start_epoch'], config['training']['epochs'] + 1):
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        generator.train()
        discriminator.train()

        for i, data in enumerate(train_loader):
            # Unpack and move data to device
            real_A, real_B = data['A'].to(device), data['B'].to(device)

            # -----------------------------
            # Train Generator
            # -----------------------------
            optimizer_G.zero_grad()
            fake_B = generator(real_A)
            pred_fake = discriminator(fake_B, real_A)
            loss_G = losses.generator_loss(pred_fake, real_B, fake_B)
            loss_G.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=max_grad_norm)
            optimizer_G.step()

            # --------------------------------
            # Train Discriminator
            # --------------------------------
            optimizer_D.zero_grad()
            # Create fresh copies of tensors to avoid in-place modifications
            real_B_copy = real_B.clone()
            real_A_copy = real_A.clone()
            fake_B_copy = fake_B.detach().clone()

            # Calculate discriminator predictions
            pred_real = discriminator(real_B_copy, real_A_copy)
            pred_fake = discriminator(fake_B_copy, real_A_copy)

            # Calculate loss
            loss_D = losses.discriminator_loss(discriminator, real_B_copy, fake_B_copy, real_A_copy)
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=max_grad_norm)
            optimizer_D.step()

            # Log losses at specified intervals
            if (i+1) % config['logging']['log_interval'] == 0:
                print(f"Step [{i+1}/{len(train_loader)}]: Generator Loss - {loss_G.item():.4f}, Discriminator Loss - {loss_D.item():.4f}")

        # Save model checkpoints at specified intervals
        if epoch % config['logging']['checkpoint_interval'] == 0:
            print(f"Saving model at epoch {epoch}")
            torch.save(generator.state_dict(), f"{config['logging']['checkpoint_dir']}/{current_model}/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"{config['logging']['checkpoint_dir']}/{current_model}/discriminator_epoch_{epoch}.pth")

        # Validation phase
        if epoch % config['logging']['validation_interval'] == 0:
            val_loss, metrics = validate_model(generator, val_loader, device, losses, epoch, config, current_model)
            
            # Update learning rates based on validation loss
            scheduler_G.step(val_loss)
            scheduler_D.step(val_loss)
            
            # Early stopping check
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    # Save the final model
    print("Saving final model...")
    torch.save(generator.state_dict(), f"{config['logging']['checkpoint_dir']}/{current_model}/generator_latest.pth")
    torch.save(discriminator.state_dict(), f"{config['logging']['checkpoint_dir']}/{current_model}/discriminator_latest.pth")
    print("Training complete.")
    
    return generator, discriminator


def validate_model(generator, val_loader, device, losses, epoch, config, current_model):
    generator.eval()
    total_l1_loss = 0
    total_ssim = 0
    total_gan_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            real_A, real_B = data['A'].to(device), data['B'].to(device)
            fake_B = generator(real_A)
            
            # Calculate various metrics
            l1_loss = losses.l1_loss(fake_B, real_B).item()
            ssim = calculate_ssim(fake_B, real_B).item()
            gan_loss = losses.gan_loss(fake_B, True).item()
            
            total_l1_loss += l1_loss
            total_ssim += ssim
            total_gan_loss += gan_loss
            num_samples += 1
            
            # Save sample validation images periodically
            if i == 0 and epoch % config['logging']['val_samples_interval'] == 0:  # Save first batch
                save_validation_samples(real_A, real_B, fake_B, epoch, f"{config['logging']['checkpoint_dir']}/{current_model}")
    
    metrics = {
        'l1_loss': total_l1_loss / num_samples,
        'ssim': total_ssim / num_samples,
        'gan_loss': total_gan_loss / num_samples
    }
    
    # Combined validation metric (weighted)
    val_loss = (metrics['l1_loss'] * 0.4 + 
                (1 - metrics['ssim']) * 0.4 + 
                metrics['gan_loss'] * 0.2)
    
    return val_loss, metrics


def save_validation_samples(real_A, real_B, fake_B, epoch, save_dir):
    """
    Save validation sample images.
    
    Args:
        real_A (Tensor): Input images
        real_B (Tensor): Ground truth images
        fake_B (Tensor): Generated images
        epoch (int): Current epoch number
        save_dir (str): Directory to save the images
    """
    import torchvision.utils as vutils
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.join(save_dir, 'validation_samples'), exist_ok=True)
    
    # Concatenate the images horizontally
    comparison = torch.cat([real_A, real_B, fake_B], dim=3)
    
    # Save the image
    vutils.save_image(
        comparison,
        os.path.join(save_dir, 'validation_samples', f'epoch_{epoch:03d}.png'),
        normalize=True
    )


if __name__ == "__main__":
    config = load_config()

    # Choose parameters
    learning_rate = 0.0003
    batch_size = 8
    lambda_l1 = 10

    config['current_training'] = {}
    config['current_training']['lr'] = learning_rate
    config['current_training']['batch_size'] = batch_size
    config['current_training']['lambda_L1'] = lambda_l1

    train_model(config)
