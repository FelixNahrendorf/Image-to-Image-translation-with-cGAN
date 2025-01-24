import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import UNetGenerator, PatchGANDiscriminator
from src.train import train_model
from src.test import test_model
from src.evaluate import evaluate_predictions, log_metrics, extract_best_median_worst, enhanced_evaluation
from utils.helper_functions import load_config
import pandas as pd
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import time


def plot_generalization_error(metrics, output_dir):
    """
    Plot generalization error metrics as violin plots.
    Args:
        metrics (dict): Dictionary of evaluation metrics.
        output_dir (str): Directory to save the plots.
    """
    data = []
    for metric, values in metrics.items():
        if isinstance(values, list):  # Only consider per-sample metrics
            for value in values:
                data.append({"Metric": metric, "Value": value})

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data)

    # Create the plot using the DataFrame
    sns.violinplot(data=df, x="Metric", y="Value")
    plt.title("Generalization Error Metrics")
    plt.savefig(os.path.join(output_dir, "generalization_error.png"))
    plt.close()


def train_apply():
    """
    Train, test, and evaluate the model with hyperparameter tuning.
    """
    # Start timing at the beginning of train_apply
    start_time = time.time()

    # Load the configuration
    config_path = "config.yaml"
    config = load_config(config_path)

    # Ensure checkpoint and logging directories exist
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Hyperparameter Tuning
    print("Starting hyperparameter tuning...")
    best_model_path = None
    best_metric = float('-inf')  # Track the best composite score

    # Reduced hyperparameter set for long training
    learning_rates = [0.0003]   # Most stable performances, anything higher becomes volatile
    batch_sizes = [8]           # More stable than 16
    lambda_l1_values = [10]     # Better balance between losses, with 25.0 it gets worse

    for lr in learning_rates:
        for batch_size in batch_sizes:
            for lambda_l1 in lambda_l1_values:
                print(f"\nTesting config: LR={lr}, Batch Size={batch_size}, Lambda L1={lambda_l1}")
                config['training']['lr'] = lr
                config['training']['batch_size'] = batch_size
                config['training']['lambda_L1'] = lambda_l1

                # Train and Test
                generator = train_model(config)
                predictions = test_model(config)

                # Evaluate
                filtered_predictions = [(real_B, fake_B) for _, real_B, fake_B in predictions]
                results = evaluate_predictions(filtered_predictions)
                log_metrics(
                    results,
                    output_file=os.path.join(checkpoint_dir, f"evaluation_results_lr{lr}_bs{batch_size}_lambda{lambda_l1}.txt"),
                )

                # Save the best model based on multiple metrics (e.g., SSIM, MSE)
                current_metric = results["SSIM"] - results["MSE"]  # Example composite metric
                if current_metric > best_metric:
                    best_metric = current_metric
                    best_model_path = os.path.join(checkpoint_dir, f"generator_lr{lr}_bs{batch_size}_lambda{lambda_l1}.pth")
                    torch.save(generator.state_dict(), best_model_path)
                    print(f"New best model saved: {best_model_path}")

    print(f"Best model achieved with Metric (Composite Score): {best_metric}")

    # Final Evaluation Phase with the Best Model
    if best_model_path is not None:
        print("Loading the best model for evaluation...")
        generator = UNetGenerator()
        discriminator = PatchGANDiscriminator()
        
        generator.load_state_dict(torch.load(best_model_path))
        discriminator.load_state_dict(torch.load(best_discriminator_path))
        
        generator.eval()
        discriminator.eval()
        
        # Run enhanced evaluation
        metrics, analysis_results = enhanced_evaluation(
            generator, 
            discriminator,
            test_loader,
            config
        )
        
        # Log results
        log_metrics(metrics, os.path.join(checkpoint_dir, "enhanced_evaluation_results.txt"))

        # Plot Generalization Error
        print("Plotting generalization error metrics for the best model...")
        plot_generalization_error(results, checkpoint_dir)

        # Showcase Test Cases
        print("Extracting best, median, and worst test cases...")
        examples = extract_best_median_worst(filtered_predictions, results)
        for case, (real_B, fake_B, score) in examples.items():
            # real_B and fake_B are currently 4D tensors (1, C, H, W)
            # We should use squeeze() to remove the batch dimension since these are individual examples
            real_img = transforms.ToPILImage()(real_B.squeeze(0))  # Remove batch dimension
            fake_img = transforms.ToPILImage()(fake_B.squeeze(0))  # Remove batch dimension
            
            real_img.save(os.path.join(checkpoint_dir, f"{case}_real.png"))
            fake_img.save(os.path.join(checkpoint_dir, f"{case}_fake.png"))
            print(f"{case.capitalize()} case saved with score: {score}")
    else:
        print("No valid model was found during hyperparameter tuning.")

    # At the end of train_apply, before the final print:
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)

    # Save timing information
    with open(os.path.join(checkpoint_dir, "training_time.txt"), 'w') as f:
        f.write(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d}\n")
        f.write(f"Total seconds: {total_time:.2f}")

    print(f"\nTotal training time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print("Train-Apply pipeline completed.")


if __name__ == "__main__":
    train_apply()
