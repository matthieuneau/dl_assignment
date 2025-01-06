from datetime import time
from tqdm import tqdm
from diffusers import (
    DDPMScheduler,
    UNet2DModel,
    EDMEulerScheduler,
    EulerAncestralDiscreteScheduler,
)
import torch
import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to(
    device
)

# Define schedulers and timesteps
schedulers = [DDPMScheduler, EDMEulerScheduler, EulerAncestralDiscreteScheduler]
timesteps_list = [5]

# Initialize plot
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle("Generated Images with Different Schedulers and Timesteps", fontsize=16)

# Loop through schedulers and timesteps
for i, scheduler in enumerate(schedulers):
    for j, timesteps in enumerate(timesteps_list):
        scheduler = scheduler.from_pretrained("google/ddpm-cat-256")
        scheduler.set_timesteps(timesteps)

        # Generate initial noise
        sample_size = model.config.sample_size
        noise = torch.randn((1, 3, sample_size, sample_size), device=device)
        input = noise

        # Generate the image
        for t in tqdm(scheduler.timesteps):
            with torch.no_grad():
                noisy_residual = model(input, t).sample
                noisy_residual = scheduler.scale_model_input(input, t)
            previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
            input = previous_noisy_sample

        # Process the image
        image = (input / 2 + 0.5).clamp(0, 1).squeeze()
        image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()

        # Plot the image in the appropriate subplot
        axes[i, j].imshow(image)
        axes[i, j].axis("off")
        axes[i, j].set_title(
            f"{scheduler.__class__}\nTimesteps: {timesteps}", fontsize=10
        )

# Adjust layout and save the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("generated_images_grid.png")
plt.show()
