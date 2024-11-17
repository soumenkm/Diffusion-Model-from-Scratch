import torch, os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from pathlib import Path
from model import DiffusionModel
from dataset import DMDataset

def load_checkpoint(device: str, checkpoint_path: Path) -> DiffusionModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DiffusionModel(device=device, image_size=checkpoint["config"]["image_size"], input_channels=checkpoint["config"]["input_channels"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to(device)
    return model

def main(device: str) -> None:
    checkpoint_path = Path(Path.cwd(), "DDPM/outputs/ckpt/DDPM_train/celebhq_1.00_1.0e-05/model_epoch_90.pt")
    model = load_checkpoint(device=device, checkpoint_path=checkpoint_path)
    ds = DMDataset(data_dir='DDPM/data/celebHQ/celeba_hq_256', image_size=256, frac=1.0)
    x_0 = ds[0].unsqueeze(0)
    model.visualize_forward_process(x_0)
    model.visualize_reverse_process()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    device = "cuda"
    main(device)