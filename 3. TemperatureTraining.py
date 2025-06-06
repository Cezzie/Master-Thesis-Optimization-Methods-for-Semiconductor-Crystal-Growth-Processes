import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
import random
import os
import wandb  # Import wandb

# Define U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        # Encoder (Downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        # Decoder (Upsampling)
        self.upconv4 = self.upconv_block(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        self.dec1 = self.conv_block(128, 64)
        # Final output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.downsample(enc1))
        enc3 = self.enc3(self.downsample(enc2))
        enc4 = self.enc4(self.downsample(enc3))
        # Bottleneck
        bottleneck = self.bottleneck(self.downsample(enc4))
        # Decoder with skip connections
        up4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat((up4, enc4), dim=1))
        up3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))
        up2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        # Final output
        return self.out_conv(dec1)

    def downsample(self, x):
        return nn.MaxPool2d(kernel_size=2)(x)


# Smaller U-Net models
class SmallUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SmallUNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.bottleneck = self.conv_block(256, 512)
        self.upconv4 = self.upconv_block(512, 256)
        self.dec4 = self.conv_block(512, 256)
        self.upconv3 = self.upconv_block(256, 128)
        self.dec3 = self.conv_block(256, 128)
        self.upconv2 = self.upconv_block(128, 64)
        self.dec2 = self.conv_block(128, 64)
        self.upconv1 = self.upconv_block(64, 32)
        self.dec1 = self.conv_block(64, 32)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))
        up4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat((up4, enc4), dim=1))
        up3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))
        up2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        return self.out_conv(dec1)

class VerySmallUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(VerySmallUNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.enc4 = self.conv_block(64, 128)
        self.bottleneck = self.conv_block(128, 256)
        self.upconv4 = self.upconv_block(256, 128)
        self.dec4 = self.conv_block(256, 128)
        self.upconv3 = self.upconv_block(128, 64)
        self.dec3 = self.conv_block(128, 64)
        self.upconv2 = self.upconv_block(64, 32)
        self.dec2 = self.conv_block(64, 32)
        self.upconv1 = self.upconv_block(32, 16)
        self.dec1 = self.conv_block(32, 16)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))
        up4 = self.upconv4(bottleneck)
        dec4 = self.dec4(torch.cat((up4, enc4), dim=1))
        up3 = self.upconv3(dec4)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))
        up2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        return self.out_conv(dec1)
    

class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(TinyUNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.bottleneck = self.conv_block(64, 128)
        self.upconv3 = self.upconv_block(128, 64)
        self.dec3 = self.conv_block(128, 64)
        self.upconv2 = self.upconv_block(64, 32)
        self.dec2 = self.conv_block(64, 32)
        self.upconv1 = self.upconv_block(32, 16)
        self.dec1 = self.conv_block(32, 16)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc3))
        up3 = self.upconv3(bottleneck)
        dec3 = self.dec3(torch.cat((up3, enc3), dim=1))
        up2 = self.upconv2(dec3)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        return self.out_conv(dec1)

class MinimalUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(MinimalUNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 16)
        self.enc2 = self.conv_block(16, 32)
        self.bottleneck = self.conv_block(32, 64)
        self.upconv2 = self.upconv_block(64, 32)
        self.dec2 = self.conv_block(64, 32)
        self.upconv1 = self.upconv_block(32, 16)
        self.dec1 = self.conv_block(32, 16)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc2))
        up2 = self.upconv2(bottleneck)
        dec2 = self.dec2(torch.cat((up2, enc2), dim=1))
        up1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat((up1, enc1), dim=1))
        return self.out_conv(dec1)
    
class TimeDependentDataset(Dataset):
    def __init__(self, data_dir, num_samples, predict_channels=(1, 2)):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.predict_channels = predict_channels  # Channels to predict (e.g., (1, 2))

        self.inputs = []
        self.targets = []

        # === Create mask once ===
        # Load the first sample just to get the shape
        first_file_path = os.path.join(data_dir, f'train_data_1_sliced.npy')
        first_data = np.load(first_file_path)
        first_input_tensor = first_data[:, :, :, 20]  # Pick t=0 or whatever makes sense

        # Create mask based on first input
        self.mask = (first_input_tensor[:, :, 0] == 0)

        # Load data from all files and extract (input, target) pairs
        for i in range(1, num_samples + 1):
            file_path = os.path.join(data_dir, f'train_data_{i}_sliced.npy')
            data = np.load(file_path)  # Expected shape: (r, z, 3, T)
            T = data.shape[3]

            for t in range(T - 1):
                if t == 1:
                    continue  # Skip timestep 1

                input_tensor = data[:, :, 1, t]  # Shape: (r, z, 1) 
                # Select the channels to predict (e.g., channel 1)
                target_tensor = data[:, :, 1, t + 1]  # Shape: (r, z, 1) or (r, z)
                
                # If we are predicting only one channel, we need to ensure target tensor has the shape (r, z, 1)
                if len(target_tensor.shape) == 2:  # If it is (r, z)
                    target_tensor = target_tensor[..., np.newaxis]  # Add a channel dimension (r, z, 1)

                # If we are using only one channel, we need to ensure inputs tensor has the shape (r, z, 1)
                if len(input_tensor.shape) == 2:  # If it is (r, z)
                    input_tensor = input_tensor[..., np.newaxis]  # Add a channel dimension (r, z, 1)


                self.inputs.append(input_tensor)
                self.targets.append(target_tensor)

        self.inputs = np.stack(self.inputs)   # Shape: (N, r, z, 3)
        self.targets = np.stack(self.targets) # Shape: (N, r, z, 1) or (N, 1, r, z)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.inputs[idx], dtype=torch.float32).permute(2, 0, 1)   # (3, r, z)
        target_tensor = torch.tensor(self.targets[idx], dtype=torch.float32).permute(2, 0, 1) # (1, r, z)

        return input_tensor, target_tensor

def make_pipeline(config):
    """Initializes the pipeline."""
    # Select model based on config
    model_type = config.get("model_type", "unet")
    
    if model_type == "small_unet":
        model = SmallUNet(in_channels=config["in_channels"], out_channels=config["out_channels"])
    elif model_type == "very_small_unet":
        model = VerySmallUNet(in_channels=config["in_channels"], out_channels=config["out_channels"])
    elif model_type == "tiny_unet":
        model = TinyUNet(in_channels=config["in_channels"], out_channels=config["out_channels"])
    elif model_type == 'minimal_unet':
        model = MinimalUNet(in_channels=config["in_channels"], out_channels=config["out_channels"])
    elif model_type =='unet':
        model = UNet(in_channels=config["in_channels"], out_channels=config["out_channels"])


        # Initialize the dataset with specific channels to predict
    dataset = TimeDependentDataset(
        data_dir=config["train_data_dir"],
        num_samples=config["num_train_samples"],
        predict_channels=config['predict_channels']  # Predict channels 1 and 2 (can change to other indices)
    )


    # Get the total number of samples in the dataset
    total_samples = len(dataset)
    
    # Calculate the size of the test set (20% of the dataset)
    test_size = int(0.2 * total_samples)
    train_size = total_samples - test_size
    
    # Perform the random split
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Initialize the data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, pin_memory=True)

    # Define loss function
    loss_fn_class = getattr(nn, config["loss_function"])
    loss_fn = loss_fn_class(**config.get("loss_params", {}))

    # Define optimizer
    optimizer = getattr(optim, config["optimizer"])(model.parameters(), lr=config["learning_rate"])

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #Mask
    mask = torch.tensor(dataset.mask, dtype=torch.bool).unsqueeze(0)
    mask = mask.unsqueeze(0).expand(config['batch_size'], -1, -1, -1)
    mask = mask.to(device)

    # Expand the mask to (batch_size, r, z)
    valid_locations = mask #~mask.unsqueeze(0).expand(config["batch_size"], *mask.shape)

    return model, train_loader, test_loader, loss_fn, optimizer, device, valid_locations

def train_one_batch(model, data, target, mask, optimizer, loss_fn, device, scaler=None):
    """Performs training on a single batch."""
    model.train()
    data, mask, target = data.to(device), mask.to(device), target.to(device)
    optimizer.zero_grad()

        # Assume batch size might change
    current_batch_size = data.shape[0]

    mask = mask[:current_batch_size]  # Take only matching batch size

    output = model(data)

    output_masked = output[mask]
    target_masked = target[mask]

    loss = loss_fn(output_masked, target_masked)

    loss.backward()
    optimizer.step()

    return loss.item()

def validate(model, loader, mask, loss_fn, device):
    """Evaluates the model on validation or test data."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

                # Assume batch size might change
            current_batch_size = data.shape[0]

            current_mask = mask[:current_batch_size]  # Take only matching batch size
            output_masked = output[current_mask]
            target_masked = target[current_mask]
            loss = loss_fn(output_masked, target_masked)

            total_loss += loss.item()

    return total_loss / len(loader)


def train(config):
    """Trains the model with early stopping and best model saving."""
    wandb.init(project=config["project_name"], config=config, name=run_name)
    model, train_loader, test_loader, loss_fn, optimizer, device, mask = make_pipeline(config)

    wandb.watch(model, loss_fn, log="all", log_freq=10)
    
    scaler = None
    
    train_losses = []
    val_losses = []

    # Early stopping parameters
    patience = config.get("early_stopping_patience", 1500)  # Stop after 10 epochs of no improvement
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(config["save_dir"], "best_model.pth")

    for epoch in range(config["epochs"]):
        # Training loop
        running_loss = 0.0
        for data, target in train_loader:
            batch_loss = train_one_batch(model, data, target, mask, optimizer, loss_fn, device, scaler)
            running_loss += batch_loss

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

        # Validation loop
        avg_val_loss = validate(model, test_loader, mask, loss_fn, device)
        val_losses.append(avg_val_loss)
        wandb.log({"epoch": epoch + 1, "val_loss": avg_val_loss})

        print(f"Epoch [{epoch+1}/{config['epochs']}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  # Reset counter

            # Save best model
            os.makedirs(config["save_dir"], exist_ok=True)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                'training_loss_history': train_losses,
                'test_loss_history': val_losses,
            }
            torch.save(checkpoint, best_model_path)
            wandb.save(best_model_path)
            print(f"Best model saved at epoch {epoch+1} with Val Loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_val_loss:.4f}")
                break  # Stop training

    # Save model and losses in a single file every few epochs
        if (epoch + 1) % config["save_frequency"] == 0:
            if not os.path.exists(config["save_dir"]):
                os.makedirs(config["save_dir"])
            save_path = os.path.join(config["save_dir"], f"checkpoint_epoch_{epoch+1}.pth")
            checkpoint = {
                "model_state_dict": model.state_dict(),
                'training_loss_history': train_losses,
                'test_loss_history': val_losses,
            }
            torch.save(checkpoint, save_path)
            wandb.save(save_path)
            
    print("Training complete!")
    wandb.finish()


def test(model, test_loader, loss_fn, device):
    """Tests the model after training."""
    test_loss = validate(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}")
    return test_loss

import matplotlib.pyplot as plt
import argparse
import sys
# Argument parser
parser = argparse.ArgumentParser(description="Train a model with configurable parameters")
parser.add_argument("--model_type", type=str, choices=["unet", "small_unet", "very_small_unet", "tiny_unet", "minimal_unet"], 
                    default="very_small_unet", help="Type of model to use")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")

args = parser.parse_args()

# Configuration with arguments
config = {
    "project_name": "New Time26April2025",
    "in_channels": 1,
    "out_channels": 1,
    "predict_channels": (1),
    "train_data_dir": 'traindatastress',
    "num_train_samples": 59,
    "batch_size": args.batch_size,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_function": "MSELoss",  # Can be changed to other loss functions, e.g., "L1Loss"
    "loss_params": {},  # Additional parameters for the loss function
    "epochs": 6000,
    "save_frequency": 3000,
    "save_dir": "saved_models_temperature_new",
    "use_amp": False,
    "model_type": args.model_type,
}

run_name = f"mt_{config['model_type']}_ts_e_{config['epochs']}_b_{config['batch_size']}"

# Configuration with arguments
config = {
    "project_name": "New Time26April2025",
    "in_channels": 1,
    "out_channels": 1,
    "predict_channels": (1),
    "train_data_dir": 'traindatastress',
    "num_train_samples": 59,
    "batch_size": args.batch_size,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "loss_function": "MSELoss",  # Can be changed to other loss functions, e.g., "L1Loss"
    "loss_params": {},  # Additional parameters for the loss function
    "epochs": 6000,
    "save_frequency": 3000,
    "save_dir": "saved_models_temperature_new" + run_name,
    "use_amp": False,
    "model_type": args.model_type,
    "early_stopping_patience": 50,
}
print(f"Training with model type: {config['model_type']} and batch size: {config['batch_size']}")

sys.stdout.flush()
# Execute training
train(config)