import os
import torch
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import load_config
from src.model import UNetGenerator, PatchGANDiscriminator
from src.evaluate import enhanced_evaluation
from data.dataset import CustomDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from src.train_apply import plot_generalization_error

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def run_analysis(config):
    """
    Run comprehensive analysis of the model and generate report data.
    This is a post-training analysis tool that uses the best model from train_apply.py
    """
    
    # Set up output directory for analysis results
    analysis_dir = os.path.join(config['logging']['checkpoint_dir'], 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() and config['device']['use_gpu'] else "cpu")
    print(f"Using device: {device}")
    
    # Load models using the latest models from train_apply.py
    print("Loading models...")
    generator = UNetGenerator()
    discriminator = PatchGANDiscriminator()
    
    # Use the same paths as in train_apply.py
    best_model_path = os.path.join(config['logging']['checkpoint_dir'], 'best_model/generator_latest.pth')
    best_discriminator_path = os.path.join(config['logging']['checkpoint_dir'], 'best_model/discriminator_latest.pth')
    
    # Load with appropriate device mapping
    generator.load_state_dict(
        torch.load(best_model_path, map_location=device)
    )
    discriminator.load_state_dict(
        torch.load(best_discriminator_path, map_location=device)
    )
    
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Prepare test dataset
    test_dataset = CustomDataset(
        images_dir=config['data']['test_images_dir'],
        labels_dir=config['data']['test_labels_dir'],
        transform=transforms.Compose([transforms.ToTensor()]),
        is_training=False
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Run enhanced evaluation
    start_time = time.time()
    metrics = enhanced_evaluation(generator, discriminator, test_loader, config)
    evaluation_time = time.time() - start_time
    
    # Convert numpy values to Python native types
    analysis_results = {
        "Performance Metrics": {
            "SSIM": float(metrics["SSIM"]),
            "MSE": float(metrics["MSE"]),
            "Average Confidence": float(metrics["Average Confidence"])
        },
        "Timing Information": {
            "Average Inference Time": float(evaluation_time / len(test_dataset)),
            "Total Evaluation Time": float(evaluation_time)
        },
        "Dataset Information": {
            "Test Set Size": len(test_dataset)
        }
    }
    
    # Save analysis results
    with open(os.path.join(analysis_dir, 'analysis_results.json'), 'w') as f:
        json.dump(analysis_results, f, indent=4, cls=NumpyEncoder)
    
    # Generate confidence distribution plot
    plt.figure(figsize=(10, 6))
    confidence_scores = [float(score) for score in metrics.get("Sample Confidence Scores", [])]
    sns.histplot(confidence_scores, bins=20)
    plt.title("Distribution of Discriminator Confidence Scores")
    plt.xlabel("Confidence Score")
    plt.ylabel("Count")
    plt.savefig(os.path.join(analysis_dir, 'confidence_distribution.pdf'))
    plt.close()
    
    # Use existing plot_generalization_error function
    plot_generalization_error(metrics, analysis_dir)
    
    # Print summary
    print("\nAnalysis Summary:")
    print("-" * 50)
    print(f"SSIM Score: {analysis_results['Performance Metrics']['SSIM']:.4f}")
    print(f"MSE Score: {analysis_results['Performance Metrics']['MSE']:.4f}")
    print(f"Average Confidence: {analysis_results['Performance Metrics']['Average Confidence']:.4f}")
    print(f"Average Inference Time: {analysis_results['Timing Information']['Average Inference Time']:.4f} seconds")
    print(f"\nResults saved in: {analysis_dir}")
    print("\nVisualization files generated:")
    print("- confidence_distribution.png")
    print("- generalization_error.png")
    print("- examples/[case]_input.png")
    print("- examples/[case]_ground_truth.png")
    print("- examples/[case]_generated.png")
    print("- attributions/[case]_integrated_gradients.png")
    print("- attributions/[case]_guided_gradcam.png")

if __name__ == "__main__":
    config = load_config()
    run_analysis(config)
