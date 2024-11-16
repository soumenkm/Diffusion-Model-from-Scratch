import os, json, pickle, torch, argparse, tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pathlib import Path
from torch.utils.data import DataLoader
import wandb
from dataset import DMDataset
from model import DiffusionModel
import numpy as np
from typing import List, Tuple, Union, Any

class DMTrainer:
    def __init__(self, device: torch.device, config: dict):
        self.config = config
        self.device = device

        self.train_ds = DMDataset(image_size=self.config["image_size"], data_dir=self.config["data_dir"], frac=self.config["frac"])
        self.dataloader = DataLoader(self.train_ds, batch_size=self.config["batch_size"], shuffle=True, num_workers=4)

        self.num_epochs = self.config["num_epochs"]
        self.project_name = "DDPM_train"
        os.environ["WANDB_PROJECT"] = self.project_name
        
        self.run_name = f"{self.config['frac']:.2f}_{self.config['initial_lr']:.1e}"
        self.model = DiffusionModel(device=self.device, image_size=self.config["image_size"], input_channels=self.config["input_channels"]).to(self.device)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['initial_lr'], weight_decay=self.config["weight_decay"], betas=self.config["adam_betas"])
   
        self.output_dir = Path(Path.cwd(), f"DDPM/outputs/ckpt/{self.project_name}/{self.run_name}")
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_checkpoint(self, epoch: int) -> None:
        checkpoint_path = Path(self.output_dir, f"model_epoch_{epoch}.pt")
        data = {"state_dict": self.model.state_dict(), "config": self.config}
        torch.save(data, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")
    
    def train(self) -> None:
        if self.config["wandb_log"]:
            wandb.init(project=self.project_name, name=self.run_name, config=self.config)
            self.num_steps = 0
        for epoch in range(1, self.num_epochs + 1):
            epoch_loss = 0.0
            with tqdm.tqdm(self.dataloader, desc=f"Epoch [{epoch}/{self.num_epochs}]") as pbar:
                for batch_idx, x_0 in enumerate(pbar):
                    x_0 = x_0.to(self.device)
                    t = torch.randint(0, self.model.forward_process.T, (x_0.size(0),), device=self.device)
                    _, loss = self.model(x_0, t)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                    pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
                    if self.config["wandb_log"]:
                        wandb.log({"Batch Loss": loss.item(), "Epoch": epoch, "Step": self.num_steps})
                        self.num_steps += 1

            if epoch % 1 == 0:
                self._save_checkpoint(epoch=epoch)
            
            avg_loss = epoch_loss / len(self.dataloader)
            print(f"Epoch [{epoch}/{self.num_epochs}] completed with Average Loss: {avg_loss:.4f}")
        
        if self.config["wandb_log"]:
            wandb.finish() 
    
def main(device: torch.device, args) -> None:
    config = {
        "image_size": args.image_size,
        "data_dir": args.data_dir,
        "input_channels": args.input_channels,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "frac": args.frac,
        "initial_lr": args.initial_lr,
        "max_grad_norm": args.max_grad_norm,
        "weight_decay": args.weight_decay,
        "adam_betas": (args.adam_beta1, args.adam_beta2),
        "grad_acc_steps": args.grad_acc_steps,
        "num_ckpt_per_epoch": args.num_ckpt_per_epoch,
        "wandb_log": args.wandb_log
    }

    trainer = DMTrainer(device=device, config=config)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model with user-specified parameters.")

    parser.add_argument("--image_size", type=int, default=256, help="Image size.")
    parser.add_argument("--data_dir", type=str, default="DDPM/data/celebHQ/celeba_hq_256", help="Data directory.")
    parser.add_argument("--input_channels", type=int, default=3, help="Number of input channels.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--frac", type=float, default=1.0, help="Fraction of data to use.")
    parser.add_argument("--initial_lr", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="Maximum gradient norm.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--adam_beta1", type=float, default=0.95, help="Beta1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Beta2 for Adam optimizer.")
    parser.add_argument("--grad_acc_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--num_ckpt_per_epoch", type=int, default=1, help="Number of checkpoints per epoch.")
    parser.add_argument("--wandb_log", type=bool, default=False, help="Enable logging to Weights and Biases.")

    args = parser.parse_args()
    print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(device=device, args=args)