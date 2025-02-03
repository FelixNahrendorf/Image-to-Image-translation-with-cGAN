import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64, dropout_rate=0.5):
        super(UNetGenerator, self).__init__()
        self.dropout_rate = dropout_rate
        
        # Encoder layers
        self.encoder = nn.ModuleList([
            self.downsample(in_channels, features),
            self.downsample(features, features * 2),
            self.downsample(features * 2, features * 4),
            self.downsample(features * 4, features * 8),
        ])
        
        # Decoder layers with skip connections
        self.decoder = nn.ModuleList([
            self.upsample(features * 8, features * 4),
            self.upsample(features * 4 * 2, features * 2),  # Double input channels for concatenated skip connections
            self.upsample(features * 2 * 2, features),       # Double input channels for concatenated skip connections
        ])

        # Final layer for output
        self.final_layer = nn.Conv2d(features * 2, out_channels, kernel_size=1)

    def downsample(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout_rate=0.5):
        """Downsampling block for U-Net encoder."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Dropout2d(dropout_rate)
        )
        
    def upsample(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout_rate=0.5):
        """Upsampling block for U-Net decoder with skip connections."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=False),
            # nn.Dropout2d(dropout_rate)
        )
        
    def forward(self, x):
        # Forward pass through encoder with debug output
        enc1 = self.encoder[0](x)
        enc2 = self.encoder[1](enc1)
        enc3 = self.encoder[2](enc2)
        enc4 = self.encoder[3](enc3)
        
        # Forward pass through decoder with skip connections and debug output
        dec1 = self.decoder[0](enc4)
        dec2 = self.decoder[1](torch.cat([dec1, enc3], dim=1))
        dec3 = self.decoder[2](torch.cat([dec2, enc2], dim=1))
        
        # Final layer with concatenated output from enc1
        dec4 = torch.cat([dec3, enc1], dim=1)
        output = self.final_layer(dec4)
        
        # Resize output to match input dimensions (256x256)
        output = torch.nn.functional.interpolate(output, size=(256, 256), mode="bilinear", align_corners=False)
        return torch.tanh(output)



class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(PatchGANDiscriminator, self).__init__()
        # We expect in_channels * 2 because we are concatenating two images
        self.model = nn.Sequential(
            self.conv_block(in_channels * 2, features, stride=2),
            self.conv_block(features, features * 2, stride=2),
            self.conv_block(features * 2, features * 4, stride=2),
            self.conv_block(features * 4, features * 8, stride=1),  # no downsampling here
        )
        self.classifier = nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)

    def conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        """Basic convolutional block for discriminator."""
        return nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=False)
        )

    def forward(self, x, y, return_features=False):
        # Check and make sure x and y have 3 channels
        if x.shape[1] != 3 or y.shape[1] != 3:
            raise ValueError(f"Expected both inputs to have 3 channels, got {x.shape[1]} and {y.shape[1]}")

        # Concatenate real/fake image and label map for conditional GAN
        combined = torch.cat([x, y], dim=1)

        features = self.model(combined)
        if return_features:
            return features

        return self.classifier(features)


def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
