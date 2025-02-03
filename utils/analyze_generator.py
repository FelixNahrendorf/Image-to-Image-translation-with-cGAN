import sys
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import UNetGenerator
from utils.helper_functions import load_config
from torchvision import transforms


def analyze_generator(config, input_dir, output_dir, method, target_layer_name, baseline=None, steps=50):
    # load generator
    generator = UNetGenerator()
    
    # Use the same paths as in train_apply.py
    best_generator_path = os.path.join(config['logging']['checkpoint_dir'], 'best_model/generator_latest.pth')
    
    device = torch.device("cuda" if config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")

    # Load with appropriate device mapping
    generator.load_state_dict(
        torch.load(best_generator_path, device, weights_only=True)
    )
    generator = generator.to(device)
    generator.eval()

    image = Image.open(input_dir).convert("RGB")
    transform = transforms.ToTensor()
    image = transform(image)

    input_image = image.to(device)

    if method:
        attributions = integrated_gradients(generator, input_image, baseline, steps)
        visualize_integrated_gradients(attributions, input_image, output_dir)
    else:
        grad_cam_visualization(generator, input_image, output_dir, target_layer_name)


def integrated_gradients(model, input_image, baseline, steps):

    if baseline is None:
        baseline = torch.zeros_like(input_image)  # Default to black baseline

    # Scale inputs from baseline to the actual input
    scaled_inputs = [
        baseline + (float(i) / steps) * (input_image - baseline) 
        for i in range(steps + 1)
    ]
    scaled_inputs = torch.stack(scaled_inputs, dim=0)

    # Compute gradients at each step
    scaled_inputs.requires_grad = True
    model.zero_grad()

    # Forward pass and backward pass to compute gradients
    output = model(scaled_inputs)
    loss = F.cross_entropy(output, torch.argmax(output, dim=1))
    loss.backward(torch.ones_like(loss), retain_graph=True)

    # Compute gradients and average over steps
    gradients = scaled_inputs.grad
    avg_gradients = gradients.mean(dim=0)

    # Compute attribution: (input - baseline) * avg_gradients
    attributions = (input_image - baseline) * avg_gradients
    return attributions

def visualize_integrated_gradients(attributions, input_image, output_dir):

    # Convert the input image to numpy for visualization
    input_image = input_image.squeeze().cpu().detach().numpy()
    input_image = np.transpose(input_image, (1, 2, 0))  # CxHxW -> HxWxC
    input_image = np.uint8(255 * input_image)

    # Convert attributions to numpy
    attributions = attributions.squeeze().cpu().detach().numpy()
    attributions = np.abs(attributions)
    attributions = attributions.sum(axis=0)
    attributions = cv2.resize(attributions, (input_image.shape[1], input_image.shape[0]))
    
    # Normalize the attributions for visualization
    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())

    # Apply a colormap to the attributions
    heatmap = np.uint8(255 * attributions)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_image = cv2.addWeighted(input_image, 0, heatmap, 1, 0)
    plt.imsave(output_dir, superimposed_image)


def grad_cam_visualization(model, input_image, output_dir, target_layer_name):
    # Ensure model is in evaluation mode
    model.eval()

    # Add batch dimension to input image
    input_tensor = input_image.unsqueeze(0)  # Shape: (1, 3, 256, 256)

    # Forward hook to get activations
    activations = {}
    def forward_hook(module, input, output):
        activations['value'] = output

    # Backward hook to get gradients
    gradients = {}
    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    # Register hooks to the target layer
    target_layer = dict(model.named_modules())[target_layer_name]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(input_tensor)

    target = output.mean()
    
    # Backward pass
    model.zero_grad()
    target.backward()

    # Get gradients and activations
    grads = gradients['value']  # Shape: (1, C, H, W)
    acts = activations['value']  # Shape: (1, C, H, W)

    # Compute Grad-CAM
    weights = grads.mean(dim=(2, 3), keepdim=True)
    grad_cam = (weights * acts).sum(dim=1, keepdim=True)
    grad_cam = F.relu(grad_cam)

    # Normalize Grad-CAM
    grad_cam = grad_cam.squeeze().detach().cpu().numpy()
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())  # Normalize to [0, 1]

    # Resize Grad-CAM to match input image size
    resize_transform = transforms.Resize((256, 256))
    grad_cam = torch.tensor(grad_cam).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    grad_cam = resize_transform(grad_cam).squeeze().numpy() 
    grad_cam = np.uint8(grad_cam * 255)
    grad_cam = np.stack([grad_cam] * 3, axis=-1)  # Convert to 3 channels for RGBg


    # Convert input image to numpy
    input_image_np = input_image.detach().cpu().numpy().transpose(1, 2, 0)  # Shape: (256, 256, 3)
    input_image_np = (input_image_np - input_image_np.min()) / (input_image_np.max() - input_image_np.min())

    # Overlay Grad-CAM heatmap on input image
    heatmap = plt.cm.jet(grad_cam[:, :, 0] / 255.0)[:, :, :3]  # Convert to RGB heatmap
    overlay = 0 * input_image_np + 1 * heatmap  # Blend heatmap and input image

    # Save overlay
    plt.imsave(output_dir, overlay)


if __name__ == "__main__":

    # Load configuration
    config_path = "config.yaml"
    config = load_config(config_path)

    # Define input directory
    input_folder = "results/prototyp1/0"

    # Define output directory
    output_folder = "results/prototyp1/7"

    # Define IG: True; Grad-CAM: False
    method = False

    # Define target layer name (Grad-CAM)
    target_layer_name = "final_layer"

    # Define steps (IG)
    steps = 50

    # loop for multiple pictures
    for i in range(1, 39):
        image_name = f"image_{i:03d}.png"
        input_dir = f"{input_folder}/{image_name}"
        output_dir = f"{output_folder}/{image_name}"
        analyze_generator(config, input_dir, output_dir, method, target_layer_name, steps=steps)
