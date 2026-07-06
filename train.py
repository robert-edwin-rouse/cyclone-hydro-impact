"""
Training script for U-Net cyclone segmentation model.
Loads pre-split data from zarr files and trains the model.
"""

import xarray as xr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import config
from model import U_Net, Trainer, init_weights
import os


# =============================================================================
# Dataset Class
# =============================================================================
class CycloneDataset(Dataset):
    """
    Pending.
    """
    def __init__(self, xarray_ds):
        """
        Pending
        """
        self.inputs = xarray_ds['inputs'].data  # Dask arrays
        self.labels = xarray_ds['precipitation'].data


    def __len__(self):
        return self.inputs.shape[0]


    def __getitem__(self, idx):
        # Compute specific time slice to load data
        x = torch.from_numpy(self.inputs[idx].compute()).float()
        y = torch.from_numpy(self.labels[idx].compute()).float()
        # Transpose to (channels, lat, lon) for CNN
        x = x.permute(2, 0, 1)
        y = y.permute(2, 0, 1)
        return x, y


# =============================================================================
# Load Data
# =============================================================================
print("Loading train, valid, test datasets from zarr...")
train_ds = xr.open_dataset(os.path.join(config.data_dir, 'train_data.zarr'),
                           engine='zarr', chunks={})
valid_ds = xr.open_dataset(os.path.join(config.data_dir, 'valid_data.zarr'),
                           engine='zarr', chunks={})
test_ds = xr.open_dataset(os.path.join(config.data_dir, 'test_data.zarr'),
                          engine='zarr', chunks={})

print(f"Train dataset shape: {train_ds['inputs'].shape}")
print(f"Valid dataset shape: {valid_ds['inputs'].shape}")
print(f"Test dataset shape: {test_ds['inputs'].shape}")


# =============================================================================
# Create PyTorch DataLoaders
# =============================================================================
print("\nCreating PyTorch DataLoaders...")
train_dataset = CycloneDataset(train_ds)
valid_dataset = CycloneDataset(valid_ds)
test_dataset = CycloneDataset(test_ds)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                          shuffle=True, num_workers=config.num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size,
                          shuffle=False, num_workers=config.num_workers)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                         shuffle=False, num_workers=config.num_workers)

print(f"Train batches: {len(train_loader)}")
print(f"Valid batches: {len(valid_loader)}")
print(f"Test batches: {len(test_loader)}")


# =============================================================================
# Initialize Model, Optimizer, Loss
# =============================================================================
print("\nInitializing model...")
model = U_Net()
model.apply(init_weights)
model.to(config.device)

print(f"Model on device: {config.device}")

optimizer = config.optimiser(model.parameters(),
                             lr=config.learning_rate,
                             weight_decay=config.weight_decay)
criterion = nn.MSELoss()


# =============================================================================
# Train Model
# =============================================================================
print(f"\nTraining for {config.epochs} epochs...")
trainer = Trainer(model, optimizer, criterion, train_loader, valid_loader)
train_losses, val_losses, val_predictions = trainer.train(config.epochs)

print("\nTraining complete!")


# =============================================================================
# Test Model
# =============================================================================
print("\nEvaluating on test set...")
model.eval()
test_losses = []
all_predictions = []
all_actuals = []

with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs.to(config.device))
        loss = criterion(outputs, targets.to(config.device))
        test_losses.append(loss.item())
        all_predictions.extend(outputs.cpu().numpy().flatten())
        all_actuals.extend(targets.cpu().numpy().flatten())

mean_test_loss = np.mean(test_losses)
print(f"Test Loss: {mean_test_loss:.6f}")

# Compute accuracy on flattened predictions
predictions_binary = np.round(np.array(all_predictions))
actuals_array = np.array(all_actuals)
accuracy = np.mean(predictions_binary == actuals_array)
print(f"Test Accuracy: {accuracy:.4f}")


# =============================================================================
# Save Model
# =============================================================================
print(f"\nSaving model to {config.model_track_path}...")
torch.save(model.state_dict(), config.model_detect_path)
print("Model saved!")
