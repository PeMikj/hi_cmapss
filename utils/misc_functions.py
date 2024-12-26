import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import config
import os


def create_time_windows(data, window_size, max_time_fraction=1.0):
    """
    Creates time windows from the data.
    """
    max_time = int(len(data) * max_time_fraction)
    windows = []
    for start in range(0, max_time - window_size + 1):
        end = start + window_size
        window = data.iloc[start:end][data.columns.difference(['unit',
         'time_cycles', 'dataset'])].values.astype(np.float32)
        windows.append(window)
    return windows


def extract_latent_vectors(autoencoder, data, window_size):
    """
    Extracts latent vectors for each data point in the unit's time series.
    :param autoencoder: Trained autoencoder model
    :param data: DataFrame for a single unit
    :param window_size: Size of the time-series windows
    :return: List of latent vectors for each time window
    """
    # Create time windows
    windows = create_time_windows(data, window_size, max_time_fraction=1.0)  # Use full data
    windows = np.array(windows)
    latent_vectors = []

    # Process each window
    for window in windows:
        # Convert to tensor
        window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        window_tensor = window_tensor.permute(0, 2, 1)  # Shape (batch_size, input_dim, window_size)

        # Encode the window
        latent_representation = autoencoder.encoder(window_tensor)
        latent_representation = latent_representation.flatten(start_dim=1)  # Flatten to (batch_size, latent_dim)

        # Store latent vector
        latent_vectors.append(latent_representation.detach().numpy().flatten())  # Convert tensor to NumPy array

    # Return all latent vectors as a NumPy array
    return np.array(latent_vectors)



def calculate_cosine_similarity_to_baseline(latent_vectors, baseline_fraction=0.1, smoothing_window=None):
    """
    Calculates cosine similarity of latent vectors relative to a baseline vector.
    """
    baseline_size = max(1, int(len(latent_vectors) * baseline_fraction))
    baseline_vector = np.mean(latent_vectors[:baseline_size], axis=0)

    if smoothing_window:
        smoothed_vectors = [
            np.mean(latent_vectors[i:i + smoothing_window], axis=0)
            for i in range(len(latent_vectors) - smoothing_window + 1)
        ]
        latent_vectors = np.array(smoothed_vectors)

    cosine_similarities = [
        cosine_similarity([baseline_vector], [vector])[0, 0]
        for vector in latent_vectors
    ]

    return cosine_similarities


def smooth_data(data, smoothing_window=5):
    """
    Smooths the data using a simple moving average.
    """
    return np.convolve(data, np.ones(smoothing_window) / smoothing_window, mode='valid')


def plot_cosine_similarity(cosine_similarity_data, output_dir='./images', unit_id=None):
    """
    Saves a plot of cosine similarity over time as an image.
    :param cosine_similarity_data: List of cosine similarity values
    :param unit_id: ID of the unit for labeling
    :param output_dir: Directory to save the images
    """
    #os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(cosine_similarity_data, label=f"Unit {unit_id}", marker='o')
    plt.title(f"Cosine Similarity Over Time (Unit {unit_id})")
    plt.xlabel("Window Index")
    plt.ylabel("Cosine Similarity")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid()

    image_path = os.path.join(output_dir, f"cosine_similarity_unit_{unit_id}.png")
    plt.savefig(image_path)
    plt.close()

    print(f"Cosine similarity plot saved to {image_path}")
