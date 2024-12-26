import torch
import torch.nn as nn

class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, window_size, latent_dim):
        """
        Simple Autoencoder for multidimensional time-series data.
        :param input_dim: int, the number of features in the time-series data.
        :param window_size: int, the length of each time-series window.
        :param latent_dim: int, the dimensionality of the latent space.
        """
        super(TimeSeriesAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc_latent = nn.Linear(window_size * 32, latent_dim)

        # Decoder
        self.fc_decode_input = nn.Linear(latent_dim, window_size * 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # Expect x with shape: (batch_size, input_dim, window_size)
        encoded = self.encoder(x)  # Output shape: (batch_size, 32, window_size)
        encoded_flat = self.flatten(encoded)  # Flatten to (batch_size, window_size * 32)
        z = self.fc_latent(encoded_flat)

        # Decoder
        decode_input = self.fc_decode_input(z)  # Shape: (batch_size, window_size * 32)
        decode_input = decode_input.view(-1, 32, x.shape[2])  # Reshape to (batch_size, 32, window_size)
        decoded = self.decoder(decode_input)
        return decoded

    def loss_function(self, recon_x, x):
        """
        Compute the reconstruction loss.
        """
        return nn.MSELoss()(recon_x, x)