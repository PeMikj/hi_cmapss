import torch
import torch.nn as nn

class VariationalTimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim, window_size, latent_dim):
        super(VariationalTimeSeriesAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(window_size * 32, latent_dim)
        self.fc_logvar = nn.Linear(window_size * 32, latent_dim)

        self.fc_decode_input = nn.Linear(latent_dim, window_size * 32)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, input_dim, kernel_size=3, stride=1, padding=1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        encoded_flat = self.flatten(encoded)
        mu = self.fc_mu(encoded_flat)
        logvar = self.fc_logvar(encoded_flat)
        z = self.reparameterize(mu, logvar)

        decode_input = self.fc_decode_input(z)
        decode_input = decode_input.view(-1, 32, x.shape[2])
        decoded = self.decoder(decode_input)
        return decoded, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = nn.MSELoss()(recon_x, x)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + kld_loss
