import os, torch, random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms

class DMDataset(Dataset):
    def __init__(self, data_dir: Path, image_size: int, frac: float):
        super(DMDataset, self).__init__()
        self.data_dir = Path(Path.cwd(), data_dir)
        self.image_size = image_size
        self.image_files = [str(f) for f in self.data_dir.iterdir() if f.name.endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(self.image_files)
        self.image_files = self.image_files[:int(frac * len(self.image_files))]
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(), # Pixel values in [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> torch.tensor:
        image_path = self.image_files[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image) # (b, C, H, W)
        except OSError:
            image = torch.zeros(3, self.image_size, self.image_size)  # Blank RGB image
        return image

if __name__ == "__main__":
    ds = DMDataset(data_dir='DDPM/data/epac/train', image_size=256, frac=1.0)
    for i in range(len(ds)):
        print(ds[i].sum())