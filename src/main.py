import os
import torch
from torch.utils.data import DataLoader
from data.dataset import LoDoPaBDataset
from models.generator import UltraLightUNetGenerator
from models.discriminator import Discriminator
from models.losses import GANLosses
from utils.training import train_gan

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    BASE_PATH = ""  # Replace with your actual path
    TRAIN_OBS_DIR = os.path.join(BASE_PATH, "observation_test")
    TRAIN_GT_DIR = os.path.join(BASE_PATH, "ground_truth_test")
    
    # Dataset and DataLoader
    dataset = LoDoPaBDataset(TRAIN_OBS_DIR, TRAIN_GT_DIR)
    dataloader = DataLoader(dataset, batch_size=12, shuffle=True, num_workers=0)
    
    # Models
    generator = UltraLightUNetGenerator().to(device)
    discriminator = Discriminator().to(device)
    
    # Loss and Optimizers
    loss_fn = GANLosses(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    
    # Training
    torch.cuda.empty_cache()
    train_gan(generator, discriminator, loss_fn, optimizer_G, optimizer_D, dataloader, epochs=5, device=device)

if __name__ == "__main__":
    main()