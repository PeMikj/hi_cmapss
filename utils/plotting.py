import matplotlib.pyplot as plt
import os
import numpy as np


def plot_losses(train_losses, val_losses, save_path):
    """
    Save training and validation losses as a plot to a specified path.

    Args:
        train_losses (list): List of training loss values.
        val_losses (list): List of validation loss values.
        save_path (str): File path to save the plot image.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")
    plt.close()


def save_autoencoder_results(results, folder_path="images"):
    """
    Save autoencoder results (original data, reconstructed data, reconstruction loss, cosine similarity) as plots.

    Args:
        results (dict): Dictionary containing autoencoder results.
        folder_path (str): Directory to save the images.
    """
    os.makedirs(folder_path, exist_ok=True)

    for subset, subset_results in results.items():
        for unit_data in subset_results:
            unit = unit_data['unit']
            subset_folder = os.path.join(folder_path, f"{subset}_unit_{unit}")
            os.makedirs(subset_folder, exist_ok=True)

            # Save original vs reconstructed data plot
            save_original_vs_reconstructed(unit_data, subset_folder)

            # Save reconstruction loss plot
            save_reconstruction_loss(unit_data, subset_folder)

            # Save cosine similarity plot
            save_cosine_similarity(unit_data, subset_folder)


def save_original_vs_reconstructed(unit_data, folder_path):
    """
    Save a plot comparing original and reconstructed data for all sensors.

    Args:
        unit_data (dict): Data for a specific unit containing original and reconstructed data.
        folder_path (str): Directory to save the plot.
    """
    original_data = unit_data['original_data']
    reconstructed_data = unit_data['reconstructed_data']
    num_sensors = original_data.shape[-1]

    for sensor_idx in range(num_sensors):
        plt.figure(figsize=(10, 5))
        timestamps = range(len(original_data))
        plt.plot(
            timestamps,
            [window[0, sensor_idx] for window in original_data],
            label="Original Data",
            alpha=0.7
        )
        plt.plot(
            timestamps,
            [window[0, sensor_idx] for window in reconstructed_data],
            label="Reconstructed Data",
            alpha=0.7
        )
        plt.title(f"Sensor {sensor_idx + 1}")
        plt.xlabel("Timestamps")
        plt.ylabel("Sensor Value")
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(folder_path, f"sensor_{sensor_idx + 1}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot for Sensor {sensor_idx + 1} to {save_path}")


def save_reconstruction_loss(unit_data, folder_path):
    """
    Save reconstruction loss plot for a unit.

    Args:
        unit_data (dict): Data for a specific unit containing reconstruction losses.
        folder_path (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(unit_data['reconstruction_losses'])
    plt.title("Reconstruction Loss")
    plt.xlabel("Timestamps")
    plt.ylabel("Loss")
    plt.grid(True)

    save_path = os.path.join(folder_path, "reconstruction_loss.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved reconstruction loss plot to {save_path}")


def save_cosine_similarity(unit_data, folder_path):
    """
    Save cosine similarity plot for a unit.

    Args:
        unit_data (dict): Data for a specific unit containing cosine similarities.
        folder_path (str): Directory to save the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(unit_data['cosine_similarities'])
    plt.title("Cosine Similarity")
    plt.xlabel("Timestamps")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)

    save_path = os.path.join(folder_path, "cosine_similarity.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved cosine similarity plot to {save_path}")
