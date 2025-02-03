import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, GuidedGradCam
import matplotlib.pyplot as plt

class XAIMethods:
    def __init__(self, generator, discriminator, device='cuda'):
        """
        Initialize XAI methods with trained models.
        
        Args:
            generator: Trained generator model
            discriminator: Trained discriminator model
            device: Device to run computations on
        """
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.generator.eval()
        self.discriminator.eval()
        
        # Initialize Captum methods
        self.integrated_gradients = IntegratedGradients(self._forward_wrapper)
        self.guided_gradcam = GuidedGradCam(self.discriminator, self.discriminator.model[-1])

    def _forward_wrapper(self, x):
        """
        Wrapper for the generator forward pass that returns a scalar value.
        We use the mean of the output as the target for attribution.
        """
        output = self.generator(x)
        return output.mean()  # Return scalar value

    def predict(self, x):
        """
        Generate prediction with confidence score.
        
        Args:
            x: Input tensor (label map)
            
        Returns:
            tuple: (prediction, confidence_score)
        """
        x = x.to(self.device)
        
        with torch.no_grad():
            # Generate fake image
            fake_B = self.generator(x)
            
            # Get discriminator score as confidence measure
            disc_score = self.discriminator(fake_B, x).mean().item()
            
            # Normalize confidence score to [0,1]
            confidence = (disc_score + 1) / 2  # Convert from [-1,1] to [0,1]
            
        return fake_B.cpu(), confidence

    def explain(self, x, method='integrated_gradients'):
        """
        Generate attribution map for the input.
        
        Args:
            x: Input tensor
            method: Attribution method ('integrated_gradients' or 'guided_gradcam')
            
        Returns:
            tensor: Attribution map
        """
        x = x.to(self.device)
        
        if method == 'integrated_gradients':
            # We don't need to specify target since we're using the wrapper
            attribution = self.integrated_gradients.attribute(x, n_steps=50)
        elif method == 'guided_gradcam':
            # Generate fake image first
            fake_B = self.generator(x)
            # Get attribution for discriminator's decision
            attribution = self.guided_gradcam.attribute(fake_B, x)
        else:
            raise ValueError(f"Unknown attribution method: {method}")
            
        return attribution.cpu()
