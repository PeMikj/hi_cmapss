import os
import pickle
import torch
from models.ae import TimeSeriesAutoencoder
from utils.misc_functions import (
    extract_latent_vectors,
    calculate_cosine_similarity_to_baseline,
    plot_cosine_similarity,
    plot_cosine_similarity_all_units,
    normalize_unit_data

)
from config import TEST_PICKLE_PATH, CHECKPOINT_DIR, WINDOW_SIZE, SELECTED_SENSORS, LATENT_DI


def load_last_checkpoint(model, checkpoint_dir):
    """
    Loads the most recent checkpoint for the model.
    """
    checkpoints = sorted(
        [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pth')],
        key=os.path.getmtime
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    last_checkpoint = checkpoints[-1]
    model.load_state_dict(torch.load(last_checkpoint))
    print(f"Loaded checkpoint: {last_checkpoint}")
    return model


if __name__ == "__main__":
    # Load test data
    with open(TEST_PICKLE_PATH, 'rb') as file:
        test_data = pickle.load(file)

    # Initialize model
    model = TimeSeriesAutoencoder(
        input_dim=len(SELECTED_SENSORS),
        window_size=WINDOW_SIZE,
        latent_dim=LATENT_DIM
    )
    model = load_last_checkpoint(model, CHECKPOINT_DIR)

    # Perform inference
    unit_id = 1  # Example: unit to analyze
    unit_data = test_data[test_data['unit'] == unit_id].sort_values(by='time_cycles')
    latent_vectors = extract_latent_vectors(model, unit_data, WINDOW_SIZE)

    # Calculate cosine similarity
    cosine_similarities = calculate_cosine_similarity_to_baseline(latent_vectors, baseline_fraction=0.1)

    # Plot results
    plot_cosine_similarity(cosine_similarities, unit_id=unit_id)
    plot_cosine_similarity_all_units(vectors)
