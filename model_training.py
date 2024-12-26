import pickle
from torch.utils.data import DataLoader
from utils.time_series_dataset import TimeSeriesDataset
from models.ae import TimeSeriesAutoencoder
from utils.training import train_vae
from utils.plotting import plot_losses
import config

if __name__ == "__main__":
    # Load training and testing data
    print("Loading pickled data...")
    with open(config.TRAIN_PICKLE_PATH, 'rb') as file:
        tr = pickle.load(file)
    with open(config.TEST_PICKLE_PATH, 'rb') as file:
        te = pickle.load(file)
    print(f"Training data shape: {tr.shape}")
    print(f"Testing data shape: {te.shape}")

    # Dataset and DataLoader setup
    dataset_tr = TimeSeriesDataset(tr, config.SELECTED_SENSORS, window_size=config.WINDOW_SIZE)
    dataloader_tr = DataLoader(dataset_tr, batch_size=config.BATCH_SIZE, shuffle=True)

    dataset_te = TimeSeriesDataset(te, config.SELECTED_SENSORS, window_size=config.WINDOW_SIZE)
    dataloader_te = DataLoader(dataset_te, batch_size=config.BATCH_SIZE, shuffle=False)

    # Model setup
    model = TimeSeriesAutoencoder(
        input_dim=len(config.SELECTED_SENSORS),
        window_size=config.WINDOW_SIZE,
        latent_dim=config.LATENT_DIM
    )

    # Train the model and save checkpoints
    train_losses, val_losses = train_vae(
        model, dataloader_tr, dataloader_te,
        num_epochs=config.NUM_EPOCHS,
        lr=config.LEARNING_RATE,
        checkpoint_dir=config.CHECKPOINT_DIR
    )

    # Plot and save training and validation losses
    plot_losses(train_losses, val_losses, save_path=config.IMAGE_SAVE_PATH)
