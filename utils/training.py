import os
import torch

def train_vae(model, dataloader_tr, dataloader_te, num_epochs, lr, checkpoint_dir):
    """
    Train the Variational Autoencoder and save model checkpoints.

    Args:
        model (nn.Module): The Variational Autoencoder model.
        dataloader_tr (DataLoader): Training DataLoader.
        dataloader_te (DataLoader): Validation DataLoader.
        num_epochs (int): Number of epochs to train.
        lr (float): Learning rate.
        checkpoint_dir (str): Directory to save model checkpoints.

    Returns:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        for batch in dataloader_tr:
            batch = batch.permute(0, 2, 1)  # For Conv1d: (batch_size, input_dim, window_size)
            reconstructed = model(batch)
            loss = model.loss_function(reconstructed, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(dataloader_tr)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in dataloader_te:
                batch = batch.permute(0, 2, 1)  # For Conv1d
                reconstructed = model(batch)
                loss = model.loss_function(reconstructed, batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(dataloader_te)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Save model checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"vae_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

    return train_losses, val_losses
