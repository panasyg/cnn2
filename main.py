import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from src.data_loader import LandmineDataset
from src.models.autoencoder import Autoencoder

device = torch.device('cpu')  # або 'cuda' при наявності GPU

def train(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x in dataloader:
        x = x.to(device)
        optimizer.zero_grad()
        x_hat = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_ssim = 0
    total_psnr = 0
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            total_loss += loss.item()

            x_np = x.squeeze().cpu().numpy()
            x_hat_np = x_hat.squeeze().cpu().numpy()

            if x_np.ndim == 3:
                x_np = x_np.mean(axis=0)
                x_hat_np = x_hat_np.mean(axis=0)

            x_np = np.clip(x_np, 0, 1)
            x_hat_np = np.clip(x_hat_np, 0, 1)

            total_ssim += ssim(x_np, x_hat_np, data_range=1.0)
            total_psnr += psnr(x_np, x_hat_np, data_range=1.0)

    num_batches = len(dataloader)
    return (
        total_loss / num_batches,
        total_psnr / num_batches,
        total_ssim / num_batches
    )

def visualize_inference(model, dataset):
    model.eval()
    sample = dataset[0].unsqueeze(0).to(device)
    with torch.no_grad():
        reconstruction = model(sample)

    input_img = sample.squeeze().cpu().numpy()
    output_img = reconstruction.squeeze().cpu().numpy()

    if input_img.ndim == 3:
        input_img = input_img.mean(axis=0)
        output_img = output_img.mean(axis=0)

    anomaly_map = (input_img - output_img) ** 2

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(input_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Reconstructed")
    plt.imshow(output_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Anomaly Map")
    plt.imshow(anomaly_map, cmap='hot')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def main(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    latent_dim = cfg['model']['latent_dim']
    hopf_cfg = cfg['model']['hopf_cfg']
    model = Autoencoder(latent_dim, hopf_cfg).to(device)

    train_dataset = LandmineDataset(cfg['data']['train_dir'])
    val_dataset = LandmineDataset(cfg['data']['val_dir'])

    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], shuffle=False, pin_memory=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg['learning_rate']),
        weight_decay=float(cfg['optimizer']['weight_decay'])
    )

    for epoch in range(cfg['epochs']):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{cfg['epochs']}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_psnr={val_psnr:.2f}, val_ssim={val_ssim:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")

    test_dataset = LandmineDataset(cfg['data']['test_dir'])
    visualize_inference(model, test_dataset)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)
