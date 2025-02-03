import torch
import torchvision.transforms as transforms
import numpy as np

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.shape) * self.std + self.mean

class FacadeAugmentation:
    def __init__(self, img_size=256):
        # All PIL-based transforms including spatial transformations
        self.transforms = transforms.Compose([
            # Spatial and color augmentations (on PIL images)
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.ColorJitter(
            #     brightness=0.1,
            #     contrast=0.1,
            #     saturation=0.1,
            # ),
            # transforms.Resize((286, 286)),
            # transforms.RandomCrop((256, 256)),
            # transforms.
            # transforms.RandomRotation(degrees=(-2.5, 2.5), fill=255),
            # transforms.RandomResizedCrop(
            #    size=(img_size, img_size),
            #     scale=(0.2, 1.0),
                # ratio=(3/4, 4/3)
            # ),
            # Convert to tensor as the last step
            transforms.ToTensor(),
            # Tensor-specific operations
            # AddGaussianNoise(mean=0., std=0.01)
        ])
        
    def __call__(self, image, mask=None):
        # Ensure inputs are PIL images
        if isinstance(image, torch.Tensor):
            raise TypeError("Expected PIL Image or numpy array, got torch Tensor")
            
        # Apply transforms with same random seed
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        image = self.transforms(image)
        
        if mask is not None:
            torch.manual_seed(seed)
            mask = self.transforms(mask)
            return image, mask
            
        return image
