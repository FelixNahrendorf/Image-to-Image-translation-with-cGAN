import torch
import torch.nn as nn

class ConditionalGANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla'):
        super(ConditionalGANLoss, self).__init__()
        if gan_mode == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy with logits
        elif gan_mode == 'lsgan':
            self.criterion = nn.MSELoss()  # LSGAN uses mean squared error
        else:
            raise NotImplementedError(f"GAN mode '{gan_mode}' not implemented")

    def forward(self, prediction, target_is_real):
        # Ensure `prediction` is a tensor
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("Prediction must be a tensor")

        # Create a target tensor based on `target_is_real`
        target_tensor = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)

        # Compute the loss
        loss = self.criterion(prediction, target_tensor)
        return loss

class GANLosses:
    def __init__(self, lambda_L1=100.0, lambda_gp=10.0, gan_mode='vanilla'):
        self.lambda_L1 = lambda_L1
        self.lambda_gp = lambda_gp
        self.gan_loss = ConditionalGANLoss(gan_mode=gan_mode)
        self.l1_loss = nn.L1Loss()

    def gradient_penalty(self, discriminator, real_samples, fake_samples, condition):
        batch_size = real_samples.size(0)
        # Create random epsilon (don't use in-place operations)
        epsilon = torch.rand(batch_size, 1, 1, 1, device=real_samples.device)
        
        # Create interpolated images (avoid in-place operations)
        interpolated = epsilon * real_samples + ((1 - epsilon) * fake_samples)
        interpolated = interpolated.requires_grad_(True)
        
        # Calculate discriminator output for interpolated images
        d_interpolated = discriminator(interpolated, condition)
        
        # Calculate gradients
        grad_outputs = torch.ones_like(d_interpolated, requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Flatten gradients
        gradients = gradients.view(batch_size, -1)
        
        # Calculate gradient penalty (avoid in-place operations)
        gradient_penalty = torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
        
        return gradient_penalty

    def generator_loss(self, pred_fake, real_image, fake_image):
        """Calculate the combined loss for the generator."""
        # GAN loss - making discriminator think the fake image is real
        loss_G_GAN = self.gan_loss(pred_fake, target_is_real=True)
        
        # L1 loss - pixel-wise similarity between fake and real images
        loss_L1 = self.l1_loss(fake_image, real_image) * self.lambda_L1
        
        # Total generator loss
        loss_G = loss_G_GAN + loss_L1
        return loss_G

    def discriminator_loss(self, discriminator, real_B, fake_B, condition):
        """Calculate the discriminator loss."""
        # Real loss
        pred_real = discriminator(real_B, condition)
        loss_D_real = self.gan_loss(pred_real, True)
        
        # Fake loss - create new tensor for fake predictions
        pred_fake = discriminator(fake_B.detach(), condition)
        loss_D_fake = self.gan_loss(pred_fake, False)
        
        # Gradient penalty
        gp = self.gradient_penalty(discriminator, real_B, fake_B, condition)
        
        # Combine losses without in-place operations
        loss_D = (loss_D_real + loss_D_fake) * 0.5 + self.lambda_gp * gp
        
        return loss_D
