import os, torch, torchinfo, tqdm
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from typing import Tuple, List, Union, Any
import torch.nn as nn
import torch.nn.functional as F
import math
from unet import UNet
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from dataset import DMDataset

class DiffusionForwardProcess:
    def __init__(self, device: str, num_steps: int, beta_start: float, beta_end: float):
        self.device = device
        self.T = num_steps
        self.beta_1 = beta_start
        self.beta_T = beta_end
        self.betas = torch.linspace(start=self.beta_1, end=self.beta_T, steps=self.T).to(self.device) # (T,)
        self.alphas = 1 - self.betas # (T,)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) # (T,)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars) # (T,)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars) # (T,)
        
    def get_noisy_image(self, t: torch.tensor, x_0: torch.tensor, noise: torch.tensor) -> torch.tensor:
        """
        q(x_t | x_0) = N(x_t | sqrt(alpha_t_bar) * x_0, (1 - alpha_t_bar) * I)
        x_t = sqrt(alpha_t_bar) * x_0 + sqrt(1 - alpha_t_bar) * epsilon
        epsilon = N(0, I)
        """
        assert x_0.shape == noise.shape, "Size of x_0, noise must match" # (b, C, H, W)
        assert x_0.ndim == 4, "The rank of x_0 should be 4"
        
        t = t.to(self.device)
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        x_t = x_0 * sqrt_alpha_bar_t + sqrt_one_minus_alpha_bar_t * noise
        return x_t # (b, C, H, W)
    
class DiffusionReverseProcess:
    def __init__(self, device: str, num_steps: int, beta_start: float, beta_end: float):
        self.device = device
        self.forward_process = DiffusionForwardProcess(device=device, num_steps=num_steps, beta_start=beta_start, beta_end=beta_end)
    
    def sample_prev_timestep(self, t: torch.tensor, x_t: torch.tensor, noise_pred: torch.tensor) -> torch.tensor:
        """
        q(x_{t-1} | x_t, x_0) = N(x_{t-1} | mu_q(x_t, t), sigma_q(t))
        p_theta(x_{t-1} | x_t, x_0) = N(x_{t-1} | mu_theta(x_t, t), sigma_q(t))
        sigma_q_t = (1-alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        x_{t-1} = mu_theta(x_t, t) * x_{t} + sqrt(sigma_q(t)) * z
        z = N(0, I)
        
        mu_q(x_t, t) = A1 * x_t + A2 * x_0
        x_0 = 1/sqrt(alpha_bar_t) * (x_t - sqrt(1 - alpha_bar_t) * epsilon_t)
        epsilon_t = N(0, I) # Ground truth noise
        mu_q(x_t, t) = A_t * x_t + B_t * epsilon_t
        epsilon_theta(x_t, t) = Unet(x_t, t) # model predicted noise
        mu_theta(x_t, t) = A_t * x_t + B_t * epsilon_theta(x_t, t)
        """
        assert x_t.shape == noise_pred.shape, "Size of x_t, noise must match" # (b, C, H, W)
        assert x_t.ndim == 4, "The rank of x_t should be 4"
        
        sqrt_one_minus_alpha_bar_t = self.forward_process.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        sqrt_alpha_t = torch.sqrt(self.forward_process.alphas[t])[:, None, None, None]
        alpha_t = self.forward_process.alphas[t][:, None, None, None]
        alpha_bar_t = self.forward_process.alpha_bars[t][:, None, None, None]
        scaling_factor = torch.clamp(sqrt_one_minus_alpha_bar_t, min=1e-10)
        mean = (x_t - ((1 - alpha_t) / (scaling_factor)) * noise_pred) / torch.clamp(sqrt_alpha_t, min=1e-10)
        if t[0].item() == 0:
            return mean 
        else:
            alpha_bar_tm1 = self.forward_process.alpha_bars[t-1][:, None, None, None]
            denominator = torch.clamp(1 - alpha_bar_t, min=1e-10)
            variance = (1 - alpha_t) * (1 - alpha_bar_tm1) / denominator
            sigma = variance ** 0.5
            z = torch.randn(x_t.shape).to(self.device)
            return mean + sigma * z
    
class DiffusionModel(nn.Module):
    def __init__(self, device: torch.device, image_size: int=256, input_channels: int=3, num_steps: int=1000, beta_start: float=1e-4, beta_end: float=0.02):
        super().__init__()
        self.image_size = image_size
        self.input_channels = input_channels
        self.unet = UNet(image_size=image_size, input_channels=input_channels)
        self.forward_process = DiffusionForwardProcess(device=device, num_steps=num_steps, beta_start=beta_start, beta_end=beta_end)
        self.reverse_process = DiffusionReverseProcess(device=device, num_steps=num_steps, beta_start=beta_start, beta_end=beta_end)
        self.device = device

    def get_latent(self, x_0: torch.tensor, t: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        noise = torch.randn_like(x_0)  # Sample Gaussian noise
        x_t = self.forward_process.get_noisy_image(t, x_0, noise)  # Get x_t using the scheduler
        return x_t, noise

    def forward(self, x_0: torch.tensor, t: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        x_t, noise = self.get_latent(x_0, t)  # Forward diffusion
        predicted_noise = self.unet(x_t, t)  # Reverse diffusion: Predict noise using UNet
        loss = F.mse_loss(predicted_noise, noise)
        return predicted_noise, loss
    
    def sample(self) -> List[torch.tensor]:
        pred_image_list = [torch.zeros(size=(1, self.input_channels, self.image_size, self.image_size)) for i in range(self.forward_process.T)]
        x_t = torch.randn((1, self.input_channels, self.image_size, self.image_size), device=self.device)
        with torch.no_grad():
            with tqdm.tqdm(reversed(range(0, self.forward_process.T)), desc="Sampling...", total=self.forward_process.T) as pbar:
                for t in pbar:
                    t_tensor = torch.tensor([t] * 1, device=self.device)
                    noise_pred = self.unet(x_t, t_tensor)
                    x_t = self.reverse_process.sample_prev_timestep(t_tensor, x_t, noise_pred)
                    pred_image_list[t] = ((torch.clamp(x_t, -1., 1.).detach().cpu() + 1)/2).squeeze(0)
        return pred_image_list
    
    def visualize_reverse_process(self, timesteps: List[int]=None) -> None:
        if timesteps is None:
            timesteps = torch.linspace(0, self.forward_process.T - 1, 16, dtype=torch.long).tolist()
        assert len(timesteps) == 16, "Provide exactly 16 timesteps for a 4x4 grid."
        
        timesteps = sorted(timesteps, reverse=True)
        samples = []
        pred_image_list = self.sample()
        for i in range(len(timesteps)):
            samples.append(pred_image_list[timesteps[i]].clone())  

        grid = make_grid(samples, nrow=4)
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        for i, t in enumerate(timesteps):
            row, col = divmod(i, 4)
            plt.text(col * self.image_size + 5, row * self.image_size + 20, f"t={t}", color="white", fontsize=12, bbox=dict(facecolor='black', alpha=0.7))

        plt.title("Reverse Process (Denoising) Visualization")
        output_path = Path(Path.cwd(), "DDPM/outputs/samples/reverse_samples.png")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        print(f"Visualization saved to {output_path}")
        
    def visualize_forward_process(self, x_0: torch.tensor, timesteps: List[int]=None) -> None:
        if timesteps is None:
            timesteps = torch.linspace(0, self.forward_process.T - 1, 16, dtype=torch.long).tolist()
        assert len(timesteps) == 16, "Provide exactly 16 timesteps for a 4x4 grid."
        
        x_0 = x_0.to(self.device)
        noise = torch.randn_like(x_0).to(self.device)
        noisy_images = []
        for t in timesteps:
            t_tensor = torch.tensor([t], device=self.device, dtype=torch.long)
            x_t = self.forward_process.get_noisy_image(t_tensor, x_0, noise)
            noisy_images.append(x_t.squeeze(0))  # Remove batch dimension

        noisy_images = [torch.clamp((img + 1) / 2, 0, 1) for img in noisy_images]
        grid = make_grid(noisy_images, nrow=4)
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        for i, t in enumerate(timesteps):
            row, col = divmod(i, 4)
            plt.text(col * x_0.size(3) + 5, row * x_0.size(2) + 20, f"t={t}", 
                    color="white", fontsize=12, bbox=dict(facecolor='black', alpha=0.7))

        plt.title("Progressive Noise Addition at Selected Timesteps")
        output_path = Path(Path.cwd(), "DDPM/outputs/samples/forward_samples.png")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        print(f"Visualization saved to {output_path}")
        
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = DiffusionModel(device="cuda").to("cuda")