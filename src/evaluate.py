from utils.metrics import structural_similarity, mean_squared_error
import numpy as np
from PIL import Image
from utils.xai_methods import XAIMethods
import torch
import os
import matplotlib.pyplot as plt

def evaluate_predictions(predictions):
    """
    Evaluate predictions using SSIM and MSE.
    Args:
        predictions (list): List of tuples (real_B, fake_B).
                            - real_B: Ground truth image/tensor.
                            - fake_B: Predicted/generated image/tensor.
    Returns:
        dict: Evaluation metrics as a dictionary.
    """
    total_ssim = []
    total_mse = []

    for real_B, fake_B in predictions:
        total_ssim.append(structural_similarity(fake_B, real_B))
        total_mse.append(mean_squared_error(fake_B, real_B))

    avg_ssim = np.mean(total_ssim)
    avg_mse = np.mean(total_mse)

    return {
        "SSIM": avg_ssim,
        "MSE": avg_mse,
        "Sample SSIM Scores": total_ssim,  # For extracting best/median/worst cases
    }

def extract_best_median_worst(predictions, metrics):
    """
    Extract the best, median, and worst test cases based on SSIM.
    Args:
        predictions (list): List of tuples (real_A, real_B, fake_B).
        metrics (dict): Dictionary of evaluation results containing per-sample SSIM scores.
    Returns:
        dict: Dictionary with best, median, and worst examples.
    """
    scores = metrics.get("Sample SSIM Scores", [])
    if not scores:
        raise ValueError("No SSIM scores found in metrics.")

    # Convert scores to numpy array for easier indexing
    scores = np.array([float(score) for score in scores])
    
    best_idx = int(np.argmax(scores))
    worst_idx = int(np.argmin(scores))
    median_idx = int(np.argsort(scores)[len(scores) // 2])

    return {
        "best": (predictions[best_idx][0], predictions[best_idx][1], float(scores[best_idx])),
        "median": (predictions[median_idx][0], predictions[median_idx][1], float(scores[median_idx])),
        "worst": (predictions[worst_idx][0], predictions[worst_idx][1], float(scores[worst_idx]))
    }

def log_metrics(results, output_file="evaluation_results.txt"):
    """
    Log evaluation metrics to a file.
    Args:
        results (dict): Evaluation metrics dictionary.
        output_file (str): File to save the metrics.
    """
    with open(output_file, "w") as file:
        file.write("Evaluation Metrics:\n")
        for metric, value in results.items():
            if isinstance(value, list):  # For per-sample metrics
                file.write(f"{metric}: List of {len(value)} values (not displayed here).\n")
            else:
                file.write(f"{metric}: {value:.4f}\n")
    print(f"Metrics logged to {output_file}")

def enhanced_evaluation(generator, discriminator, test_loader, config):
    """
    Perform enhanced evaluation including XAI analysis.
    """
    device = torch.device("cuda" if config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")
    xai = XAIMethods(generator, discriminator, device)
    
    # Standard evaluation metrics
    predictions = []
    confidences = []
    
    print("Running enhanced evaluation...")
    for batch in test_loader:
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)
        
        try:
            # Get prediction and confidence
            fake_B, confidence = xai.predict(real_A)
            predictions.append((real_A.cpu(), real_B.cpu(), fake_B))
            confidences.append(confidence)
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            continue
    
    # Calculate standard metrics
    metrics = evaluate_predictions([(p[1], p[2]) for p in predictions])
    metrics['Average Confidence'] = np.mean(confidences)
    metrics['Sample Confidence Scores'] = confidences
    
    # Create visualization directories
    vis_dir = os.path.join(config['logging']['checkpoint_dir'], 'visualizations')
    examples_dir = os.path.join(vis_dir, 'examples')
    attribution_dir = os.path.join(vis_dir, 'attributions')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(examples_dir, exist_ok=True)
    os.makedirs(attribution_dir, exist_ok=True)
    
    print(f"\nSaving visualizations to: {vis_dir}")
    
    # Generate explanations for best/worst cases
    best_worst = extract_best_median_worst(predictions, metrics)
    for case_name, (real_A, real_B, score) in best_worst.items():
        try:
            # Save input, ground truth, and generated images
            save_tensor_as_image(real_A.squeeze(0), os.path.join(examples_dir, f'{case_name}_input.png'))
            save_tensor_as_image(real_B.squeeze(0), os.path.join(examples_dir, f'{case_name}_ground_truth.png'))
            
            # Generate and save predictions
            fake_B, _ = xai.predict(real_A)
            save_tensor_as_image(fake_B.squeeze(0), os.path.join(examples_dir, f'{case_name}_generated.png'))
            
            print(f"\nGenerating explanations for {case_name} case...")
            # Generate explanations
            ig_attr = xai.explain(real_A, method='integrated_gradients')
            gradcam_attr = xai.explain(real_A, method='guided_gradcam')
            
            # Save attributions
            save_attribution_visualization(
                ig_attr, 
                os.path.join(attribution_dir, f'{case_name}_integrated_gradients.png')
            )
            save_attribution_visualization(
                gradcam_attr, 
                os.path.join(attribution_dir, f'{case_name}_guided_gradcam.png')
            )
            print(f"Saved {case_name} case visualizations")
            
        except Exception as e:
            print(f"Error generating explanations for {case_name} case: {str(e)}")
            continue
    
    return metrics

def save_tensor_as_image(tensor, path):
    """Save a tensor as an image."""
    if tensor.dim() == 4:  # If tensor has batch dimension
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3:  # If tensor is in CxHxW format
        tensor = tensor.permute(1, 2, 0)  # Convert to HxWxC
    
    # Convert to numpy and ensure values are in [0, 1]
    img_np = tensor.detach().cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_np = (img_np * 255).astype(np.uint8)
    
    # Save image
    plt.imsave(path, img_np)

def save_attribution_visualization(attribution, save_path):
    """
    Save attribution map visualization.
    """
    # Normalize attribution values
    attr_norm = (attribution - attribution.min()) / (attribution.max() - attribution.min())
    
    # Convert to heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(attr_norm.mean(dim=0), cmap='hot')
    plt.colorbar()
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()
