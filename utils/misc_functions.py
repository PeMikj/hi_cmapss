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

# utils/misc_functions.py

def normalize_unit_data(vectors_data):
    """
    Normalizes the cosine_similarities for each unit in the given vectors data.
    :param vectors_data: List of unit data, where each unit has 'cosine_similarities'.
    :return: The updated list with normalized cosine similarities.
    """
    for unit_data in vectors_data:
        similarities = unit_data['cosine_similarities']
        min_val = min(similarities)
        max_val = max(similarities)

        # Avoid division by zero if all values are the same
        if max_val == min_val:
            unit_data['cosine_similarities'] = [0.5] * len(similarities)  # Assign 0.5 if all values are identical
        else:
            unit_data['cosine_similarities'] = [
                (x - min_val) / (max_val - min_val) for x in similarities
            ]
    return vectors_data

def plot_cosine_similarity_all_units(cosine_similarity_data, units_to_plot=None):
    """
    Plots cosine similarity relative to a baseline vector over time for all units on a single plot.
    :param cosine_similarity_data: Dictionary with subset names as keys and cosine similarity data for each unit
    :param units_to_plot: List of specific unit IDs to plot (optional). If None, plots all units.
    """
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20.colors  # Use a colormap for distinct unit colors
    color_idx = 0

    for subset, subset_cosine_similarities in cosine_similarity_data.items():
        normalized_data = normalize_unit_data(subset_cosine_similarities)
        for unit_data in normalized_data:
            unit_id = unit_data['unit']
            cosine_similarities = unit_data['cosine_similarities']

            # Filter specific units to plot if specified
            if units_to_plot and unit_id not in units_to_plot:
                continue

            # Plot cosine similarity for the current unit
            plt.plot(
                cosine_similarities, 
                label=f'Unit {unit_id} ({subset})', 
                marker='o', 
                color=colors[color_idx % len(colors)]
            )
            color_idx += 1

    # Plot settings
    plt.title("Cosine Similarity to Baseline Over Time (All Units)")
    plt.xlabel("Time Step")
    plt.ylabel("Cosine Similarity (Normalized)")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()