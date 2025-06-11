import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import wandb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime


# Smaller U-Net models, note that only small U-Net class is loaded here since we got the best performing models being small
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
    
class HeaterOptimization(nn.Module):
    def __init__(self, model_temp, model_stress, rollout_length, device):
        super(HeaterOptimization, self).__init__()
        self.model_temp = model_temp
        self.model_stress = model_stress
        self.rollout_length = rollout_length
        self.device = device

    def forward(self, initial_temp, mask, initial_guess=None, num_epochs=500, 
                penalty_weight_max_temp=0.0, penalty_weight_smoothness = 1.0, penalty_weight_stress = 0.0, penalty_weight_decreasing_temp = 0.0):
        """
        Run a rollout of the temperature model with input heaters and compute stress loss.
        
        Args:
        - initial_temp (Tensor): Starting temperature field (shape: (r, z)).
        - mask (Tensor): Heater mask (where the heaters are active, shape: (r, z)).
        - initial_guess (Tensor, optional): Initial guess for the heaters, shape (rollout_length, r).
        - num_epochs (int, optional): Number of optimization epochs.
        - penalty_weight (float, optional): Weight of the penalty term for the max temperature constraint.
        
        Returns:
        - heaters (Tensor): Optimized heater configuration for each timestep (shape: (rollout_length, r)).
        """
        s0, s1 = initial_temp.shape  # (r, z)

        # If no initial guess is provided, start with zeros
        if initial_guess is None:
            print("starting from no initial guess")
            heaters = torch.zeros(self.rollout_length, s0, device=self.device, requires_grad=True)
        else:
            print("using initial guess")
            # If an initial guess is provided, use it to initialize the heaters
            heaters = initial_guess.clone().detach().to(self.device).requires_grad_(True)
            print("heaters shape", heaters.shape)
            print(config["initial_guess"])
        optimizer = optim.Adam([heaters], lr=0.1)  # You can adjust the learning rate

        prev_max_temp = None  # Initialize previous max temperature for comparison

        for step in range(num_epochs):  # Adjust steps for optimization iterations
            optimizer.zero_grad()

            # Start with the initial temperature
            input_tensor = initial_temp.clone().unsqueeze(0).unsqueeze(0)  # shape (1,1,r,z)
            rollout = []
            total_loss = 0.0
            total_loss_maxtemp = 0.0
            total_loss_smoothness = 0.0
            total_loss_decreasing_temp = 0.0

            # 1. Run the temperature model over all time steps
            for t in range(self.rollout_length):
                # Insert heater values into input (only at masked locations)
                input_tensor[0, 0][~mask] = heaters[t]

                #Max temperature loss term added
                loss_maxtemp = heaters[t].max() 

                # Second-order differences
                diffs = heaters[t][2:] - 2 * heaters[t][1:-1] + heaters[t][:-2]  # shape: [94]

                # Smoothness penalty as standard deviation of the differences
                # Total variation regularization
                loss_smoothness = torch.mean(diffs ** 2)
                
                loss_decreasing_temp = 0
                # After the first timestep we can add a penalty term 
                if t != 0:
                    
                    # heater: shape [96]
                    diffs_temp = torch.mean(heaters[t]) - torch.mean(heaters[t-1])  # shape [95]
                    # print("mean version")
                    # Only penalize positive changes (i.e., increases)
                    loss_decreasing_temp = torch.clamp(diffs_temp, min=0)

                pred_temp = self.model_temp(input_tensor)  # shape (1,1,r,z)

                rollout.append(pred_temp[0])  # Remove batch dim for stacking
                input_tensor = pred_temp  # Update input tensor for next timestep

                
                total_loss_maxtemp += loss_maxtemp
                total_loss_smoothness += loss_smoothness
                total_loss_decreasing_temp += loss_decreasing_temp

            # Shape: (rollout_length, 1, r, z)
            rollout_tensor = torch.stack(rollout)

            # 2. Predict stress using the stress model
            pred_stress = self.model_stress(rollout_tensor)

            # 3. Compute the stress loss (mean stress across the entire rollout)
            total_loss_stress = pred_stress.max()

            
            

            # Add the other losses with weight
            total_loss += total_loss_stress * penalty_weight_stress

            total_loss += total_loss_maxtemp * penalty_weight_max_temp

            total_loss += total_loss_smoothness * penalty_weight_smoothness

            total_loss += total_loss_decreasing_temp * penalty_weight_decreasing_temp

            # print(loss)
            T_max = 1511  # for example

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                heaters.clamp_(min=0.0, max=T_max)  # Replace T_max with your desired maximum temperature


            # # Update the previous max temperature for the next timestep
            # prev_max_temp = rollout_tensor.max(dim=2)[0].max(dim=2)[0].detach().item()  # Get max for the next timestep

            # Print loss for monitoring
            if step % 10 == 0:
                wandb.log({
                    "loss": total_loss.item(),
                    "stress_loss": total_loss_stress.item(),
                    "step": step,
                    "max_heater": heaters.max().item(),
                    "mean_heater": heaters.mean().item(),
                })
                print(
                    f"[Step {step}] Loss: {total_loss.item():.5f}, "
                    f"Loss Stress: {total_loss_stress.item():.5f}, "
                    f"Loss Temp: {total_loss_maxtemp.item():.5f}, "
                    f"Loss Smoothness: {total_loss_smoothness.item():.5f}, "
                    f"Loss Decreasing Temp: {total_loss_decreasing_temp.item():.5f}"
                )

        return heaters.detach(), rollout_tensor, pred_stress # Return the optimized heaters, and the final temp fields and stress


import argparse
import sys

# Argument parser
parser = argparse.ArgumentParser(description="Optimize Heater Configurations")
parser.add_argument("--start_pos", type=int, default=2, help="starting position in the 20 Hours")
parser.add_argument("--epochs", type=int, default=10000, help="Epochs")
parser.add_argument("--rollout_length", type=int, default=10, help="length of rollout")
parser.add_argument("--penalty_weight_max_temp", type=float, default=0.0, help="loss weight mt")
parser.add_argument("--penalty_weight_smoothing", type=float, default=0.0, help="loss weight sm")
parser.add_argument("--penalty_weight_decreasing_temp", type=float, default=0.0, help="loss weight dt")
parser.add_argument("--penalty_weight_stress", type=float, default=1.0, help="loss weight s")
parser.add_argument("--initial_guess", type=str, default= None, help= "Initial guess file")
args = parser.parse_args()


# --- Config ---
config = {
    "project_name": "ORs",
    "train_data_dir": 'traindatastress',
    "temp_model": "checkpoint_epoch_6000_temperature_small.pth",
    "stress_model": "best_model_stress_small.pth",
    "test_data": 60,
    "optimizer": "Adam",
    "epochs": args.epochs,
    "save_frequency": 6000,
    "save_dir": "models",
    "start_pos": args.start_pos,
    "rollout_length": args.rollout_length,
    "penalty_weight_max_temp": args.penalty_weight_max_temp,
    "penalty_weight_smoothness": args.penalty_weight_smoothing,
    "penalty_weight_decreasing_temp": args.penalty_weight_decreasing_temp,
    "penalty_weight_stress": args.penalty_weight_stress,
    "initial_guess": args.initial_guess
}

# --- WandB Init ---
wandb.init(project=config["project_name"], config=config)

# --- UNet and Optimization Classes (same as before, no change needed) ---
# [Paste your SmallUNet and HeaterOptimization definitions here unchanged...]

# Load models
model_temp = SmallUNet(in_channels=1, out_channels=1)
model_stress = SmallUNet(in_channels=1, out_channels=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_temp = model_temp.to(device)
model_stress = model_stress.to(device)

checkpoint = torch.load(os.path.join(config["save_dir"], config["temp_model"]), map_location=device)
model_temp.load_state_dict(checkpoint["model_state_dict"])

checkpoint_stress = torch.load(os.path.join(config["save_dir"], config["stress_model"]), map_location=device)
model_stress.load_state_dict(checkpoint_stress["model_state_dict"])

model_temp.eval()
model_stress.eval()

t_start = config["start_pos"]
heater_optimizer = HeaterOptimization(model_temp, model_stress, rollout_length=config["rollout_length"], device=device)

test_data = np.load(os.path.join(config["train_data_dir"], f"train_data_{config['test_data']}_sliced.npy"))
initial_temp = torch.tensor(test_data[:, :, 1, t_start], dtype=torch.float32).to(device)
# Initial guess for heaters, if initial guess is given we assume this to be a .pt file
# Otherwise the initial guess will just be the test data sample
mask = (test_data[:, :, 0, t_start] == 0)
if config["initial_guess"]:
    initial_guess = torch.load(os.path.join(config["save_dir"], config["initial_guess"]))
    
else:
    initial_guess = torch.tensor(
    test_data[:, :, 0, t_start:t_start + config["rollout_length"]][~mask],
    dtype=torch.float32
    ).transpose(0, 1).to(device)



print("mask shape:", mask.shape)
print("intial guess shape", initial_guess.shape)


optimized_heaters, temperature_ros, stress_ros = heater_optimizer(initial_temp, mask, 
                                                                  initial_guess=initial_guess, 
                                                                  num_epochs=config["epochs"],
                                                                  penalty_weight_max_temp=config["penalty_weight_max_temp"], 
                                                                  penalty_weight_smoothness = config["penalty_weight_smoothness"],
                                                                  penalty_weight_stress = config["penalty_weight_stress"], 
                                                                  penalty_weight_decreasing_temp = config["penalty_weight_decreasing_temp"])

# --- Save optimized heaters tensor ---
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"optimized_heaters_{config['test_data']}_{timestamp}.pt"
save_path = os.path.join(config["save_dir"], filename)
torch.save(optimized_heaters.cpu(), save_path)
print(f"Saved optimized heaters to {save_path}")

# Optionally log tensor to wandb
artifact = wandb.Artifact(f"optimized_heaters_{config['test_data']}", type="tensor")
artifact.add_file(save_path)
wandb.log_artifact(artifact)

# --- Plotting and Logging to wandb ---

# Assuming optimized_heaters is a PyTorch tensor of shape (10, 96)
max_values_per_timestep, _ = optimized_heaters.max(dim=1)

# Get the max value for each timestep in the initial guess
max_values_initial_guess, _ = initial_guess.max(dim=1)

# Max heater values
fig1, ax1 = plt.subplots()
ax1.plot(max_values_per_timestep.cpu(), label='Optimized')
ax1.plot(max_values_initial_guess.cpu(), label='Initial Guess', linestyle='--')
ax1.set_xlabel('Timestep')
ax1.set_ylabel('Max Heater Value')
ax1.set_title('Max Heater Value at Each Timestep')
ax1.legend()
wandb.log({"Max Heater Plot": wandb.Image(fig1)})

# Assuming optimized_heaters is a PyTorch tensor of shape (10, 96)
max_values_per_timestep = optimized_heaters.mean(dim=1)

# Get the max value for each timestep in the initial guess
max_values_initial_guess = initial_guess.mean(dim=1)

# Mean heater values
fig2, ax2 = plt.subplots()
ax2.plot(max_values_per_timestep.cpu(), label='Optimized')
ax2.plot(max_values_initial_guess.cpu(), label='Initial Guess', linestyle='--')
ax2.set_xlabel('Timestep')
ax2.set_ylabel('Mean Heater Value')
ax2.set_title('Mean Heater Value at Each Timestep')
ax2.legend()
wandb.log({"Mean Heater Plot": wandb.Image(fig2)})
plt.close(fig1)
# Heater at t=0
fig3, ax3 = plt.subplots()
ax3.plot(initial_guess[0].cpu(), label='Initial at t=0')
ax3.plot(optimized_heaters[0].cpu(), label='Heater at t=0')
ax3.set_xlabel('z')
ax3.set_ylabel('K')
ax3.set_title('Optimized Heaters Over Time')
ax3.legend()
wandb.log({"Heater Profile t=0": wandb.Image(fig3)})

# Heater at t=4
fig4, ax4 = plt.subplots()
ax4.plot(initial_guess[4].cpu(), label='Initial at t=4')
ax4.plot(optimized_heaters[4].cpu(), label='Heater at t=4')
ax4.set_xlabel('z')
ax4.set_ylabel('K')
ax4.set_title('Optimized Heaters Over Time')
ax4.legend()
wandb.log({"Heater Profile t=4": wandb.Image(fig4)})
plt.close(fig1)
plt.close(fig2)
plt.close(fig3)
plt.close(fig4)

stress_ros = stress_ros.detach().cpu()

min_val = -40775.78581437509
max_val = 27578406.50749634

# Denormalize
stress_denorm = stress_ros * (max_val - min_val) + min_val

temperature_ros = temperature_ros.detach().cpu()

_, _, z_dim,r_dim = stress_ros.shape

plotmask = temperature_ros <= 273

# Mask the temperature field for positive values
masked_stress = np.ma.masked_where(plotmask, stress_denorm)

# Set vmin and vmax from the masked (positive) values
vminstress = masked_stress.min()
vmaxstress = masked_stress.max()

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(4, 6), constrained_layout=True)
# fig.tight_layout()
cax = ax.imshow(
    np.zeros((r_dim, z_dim)),  # Initial empty frame
    cmap='viridis',
    origin='lower',
    extent=[0, 0.06, 0, 0.15],
    vmin=vminstress,  # Set consistent scale
    vmax=vmaxstress
)
cbar = fig.colorbar(cax, ax=ax, label="Pa")
ax.set_title("Stress Field")
ax.set_xlabel("r (m)")
ax.set_ylabel("z (m)")

# Update function for each frame
def update(t):
    masked_data = np.ma.masked_where(~mask, masked_stress[t,0])
    cax.set_array(masked_data)
    ax.set_title(f"Stress Field, n = {t}")
    return [cax]

# Create the animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=config["rollout_length"]
)

frps = config["rollout_length"]//5

filename_stress = f"stress_video_{config['test_data']}_{timestamp}.gif"
save_path_stress_vid = os.path.join(config["save_dir"], filename_stress)
ani.save(save_path_stress_vid, fps=frps)
wandb.log({"Stress Animation": wandb.Video(save_path_stress_vid, format="gif")})
plt.close(fig)

# Mask the temperature field for positive values
masked_temp = np.ma.masked_where(plotmask, temperature_ros)

broadcasted_mask = mask[np.newaxis, np.newaxis, ...]  # shape (1, 1, r, z)
broadcasted_mask = np.broadcast_to(broadcasted_mask, temperature_ros.shape)  # (T, 1, r, z)
# Set vmin and vmax from the masked (positive) values
vmintemp = masked_temp[~broadcasted_mask].mean() - 200
vmaxtemp = masked_temp[~broadcasted_mask].max()
print(vmintemp)
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(4, 6), constrained_layout=True)
# fig.tight_layout()
cax = ax.imshow(
    np.zeros((r_dim, z_dim)),  # Initial empty frame
    cmap='viridis',
    origin='lower',
    extent=[0, 0.06, 0, 0.15],
    vmin=vmintemp,  # Set consistent scale
    vmax=vmaxtemp
)
cbar = fig.colorbar(cax, ax=ax, label="Pa")
ax.set_title("Temperature Field")
ax.set_xlabel("r (m)")
ax.set_ylabel("z (m)")

# Update function for each frame
def update(t):
    masked_data = np.ma.masked_where(~mask, masked_temp[t,0])
    cax.set_array(masked_data)
    ax.set_title(f"Temperature Field, n = {t}")
    return [cax]

# Create the animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=config["rollout_length"]
)

frps = config["rollout_length"]//5

filename_temp = f"temp_video_{config['test_data']}_{timestamp}.gif"
save_path_temp_vid = os.path.join(config["save_dir"], filename_temp)
ani.save(save_path_temp_vid, fps=frps)
wandb.log({"Temp Animation": wandb.Video(save_path_temp_vid, format="gif")})
plt.close(fig)


# Remove the singleton dim -> shape: (T, r, z)
stress = stress_denorm[:, 0, :, :]  # or stress_ros.squeeze(1)

# Compute max stress per timestep (flatten r and z)
max_stress_per_timestep = stress.reshape(stress.shape[0], -1).max(axis=1)

# Plot
fig_s, ax = plt.subplots()
ax.plot(max_stress_per_timestep.values)
ax.set_xlabel("Timestep")
ax.set_ylabel("Max Stress (Pa)")
ax.set_title("Max Stress per Timestep")
ax.grid(True)
wandb.log({"Max Stress per Timestep": wandb.Image(fig_s)})
plt.close(fig_s)
# Convert to NumPy if needed
initial = initial_guess.cpu().numpy()
optimized = optimized_heaters.cpu().numpy()

T, Z = optimized.shape

fig, ax = plt.subplots()
# Lighter, thinner, dashed line for the initial guess
line1, = ax.plot([], [], label='Initial', color='lightgray', linestyle='--', linewidth=1.0)

# Stronger color for the optimized heater
line2, = ax.plot([], [], label='Optimized', color='orange', linewidth=1.0)

ax.set_xlim(0, Z)
ax.set_ylim(
    min(initial.min(), optimized.min()) - 5,
    max(initial.max(), optimized.max()) + 5
)
ax.set_xlabel('z')
ax.set_ylabel('K')
ax.set_title('Heater Profile Over Time')
ax.legend()

def update(t):
    line1.set_data(range(Z), initial[t])
    line2.set_data(range(Z), optimized[t])
    ax.set_title(f"Heater Profile — Timestep {t}")
    return [line1, line2]

ani = animation.FuncAnimation(
    fig, update, frames=T, interval=100, blit=True
)

# Save as .gif
filename_heater = f"heater_animation_{timestamp}.gif"
save_path_heater = os.path.join(config["save_dir"], filename_heater)
ani.save(save_path_heater, fps=frps)  # Adjust fps as needed
wandb.log({"Heater Animation": wandb.Video(save_path_heater, format="gif")})

# Plot
# --- Finish WandB run ---
wandb.finish()
