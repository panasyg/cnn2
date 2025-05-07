# evaluate.py

import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_loader import LandmineDataset
from src.models.autoencoder import Autoencoder

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            recon = model(x)
            loss = criterion(recon, x)
            running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

def main(config_path: str, weights_path: str):
    # Load config
    cfg = load_config(config_path)
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset & DataLoader
    test_ds = LandmineDataset(cfg['data']['test_dir'])
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    # Model
    model = Autoencoder(
        latent_dim=cfg['model']['latent_dim'],
        hopf_cfg=cfg['model']
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # Criterion
    criterion = nn.MSELoss()

    # Evaluate
    test_loss = eval_epoch(model, test_loader, criterion, device)
    print(f"Test MSE Loss: {test_loss:.6f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate trained autoencoder on test set")
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to config file (YAML)'
    )
    parser.add_argument(
        '--weights', type=str, default='best_model.pth',
        help='Path to model weights (.pth)'
    )
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    main(args.config, args.weights)
