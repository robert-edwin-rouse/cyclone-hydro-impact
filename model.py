"""
This module defines the neural network architecture for the cyclone tracking
algorithm.  It uses a U-Net architecture to segment the 
"""

import torch
import torch.nn as nn
import numpy as np
import config


# =============================================================================
# U-Net Architecture Components
# =============================================================================
class convolution_block(nn.Module):
    """
    Basic convolutional block used in the U-Net architecture, consisting of 
    two consecutive convolutional and batch normalization layers followed by 
    SiLU activation.
    """

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Conv2d(output_channels, output_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class downsampling_block(nn.Module):
    """
    A downsampling block for the U-Net contracting path, consisting of a 
    convolutional block followed by a max pooling operation.
    """

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv_block = convolution_block(input_channels, output_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.conv_block(x)
        p = self.pool(x)
        return x, p


class upsampling_block(nn.Module):
    """
    An upsampling block for the U-Net expansive path, consisting of an 
    up-convolution followed by a convolution block.
    """

    def __init__(self, input_channels, output_channels):
       super().__init__()
       self.up_convolution = nn.ConvTranspose2d(
           input_channels, output_channels, kernel_size=2, stride=2, padding=0)
       self.convolution_block = convolution_block(
           output_channels + output_channels,
           output_channels
       )

    def forward(self, x, skip_connection):
        x = self.up_convolution(x)
        skip_connection = skip_connection[:, :, :x.shape[2], :x.shape[3]]
        x = torch.cat([x, skip_connection], axis=1)
        x = self.convolution_block(x)
        return x


class U_Net(nn.Module):
    """
    The U-Net architecture for cyclone tracking, consisting of a contracting 
    path (encoder), a bottleneck, and an expansive path (decoder).
    """

    def __init__(self):
        super().__init__()
        self.encoder_1 = downsampling_block(15, 64)
        self.encoder_2 = downsampling_block(64, 128)
        self.encoder_3 = downsampling_block(128, 256)
        self.encoder_4 = downsampling_block(256, 512)

        self.bottleneck = convolution_block(512, 1024)

        self.decoder_1 = upsampling_block(1024, 512)
        self.decoder_2 = upsampling_block(512, 256)
        self.decoder_3 = upsampling_block(256, 128)
        self.decoder_4 = upsampling_block(128, 64)

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.final_upsample = nn.Upsample(size=(46, 46), mode='bilinear', align_corners=False)

    def forward(self, x):
        x1, p1 = self.encoder_1(x)
        x2, p2 = self.encoder_2(p1)
        x3, p3 = self.encoder_3(p2)
        x4, p4 = self.encoder_4(p3)

        b = self.bottleneck(p4)

        d1 = self.decoder_1(b, x4)
        d2 = self.decoder_2(d1, x3)
        d3 = self.decoder_3(d2, x2)
        d4 = self.decoder_4(d3, x1)

        output = self.output_layer(d4)
        output = self.final_upsample(output)
        
        return output


def init_weights(m: nn.Module):
    """
    Randomly initialises weights in the neural network model.
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)


# =============================================================================
# Training
# =============================================================================
class Trainer:
    """
    Trains and validates the model.
    """

    def __init__(self, model, optimiser, criterion, train_loader, val_loader):
        self.model = model
        self.optimiser = optimiser
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_val_loss = float('inf')

    def training_pass(self):
        """
        Single epoch training loop.
        """
        self.model.train()
        losses = []
        for inputs, targets in self.train_loader:
            self.optimiser.zero_grad()
            outputs = self.model(inputs.to(config.device))
            # Handle both single value and multi-value targets
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            loss = self.criterion(outputs, targets.to(config.device))
            loss.backward()
            self.optimiser.step()
            losses.append(loss.item())
        return np.mean(losses)

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluate on validation set.
        """
        self.model.eval()
        losses, predictions, actuals = [], [], []
        for inputs, targets in self.val_loader:
            outputs = self.model(inputs.to(config.device))
            # Handle both single value and multi-value targets
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            loss = self.criterion(outputs, targets.to(config.device))
            losses.append(loss.item())

            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(targets.cpu().numpy().flatten())
        return np.mean(losses), predictions, actuals

    def train(self, num_epochs):
        """
        Train the model for the specified number of epochs.
        """
        train_losses, val_losses = [], []
        for epoch in range(1, num_epochs+1):
            train_loss = self.training_pass()
            train_losses.append(train_loss)
            val_loss, val_predictions, val_actuals = self.evaluate()
            val_losses.append(val_loss)
            print(f'Epoch {epoch}/{num_epochs} -> '
                f'Training Loss: {train_loss:.6f}, '
                f'Validation Loss: {val_loss:.6f}')
        return train_losses, val_losses, (val_predictions, val_actuals)
