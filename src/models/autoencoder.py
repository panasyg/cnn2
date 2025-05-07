import torch
import torch.nn as nn
from src.models.hopf_conv import HopfOscillator3D


class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int, hopf_cfg: dict):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.enc2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool3d((1, 2, 2))
        # Hopf layer
        self.hopf = HopfOscillator3D(units=hopf_cfg['hopf_units'],
                                     dt=hopf_cfg['hopf_dt'],
                                     alpha=hopf_cfg['hopf_alpha'],
                                     beta=hopf_cfg['hopf_beta'])
        # Bottleneck
        self.bottleneck = nn.Conv3d(64, latent_dim, kernel_size=1)
        # Decoder
        self.dec1 = nn.ConvTranspose3d(latent_dim, 64, kernel_size=3, stride=1, padding=1)
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.dec2 = nn.ConvTranspose3d(64, 32, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.out = nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.enc1(x))
        x = self.pool1(x)
        x = torch.relu(self.enc2(x))
        x = self.pool2(x)
        x = self.hopf(x)
        x = torch.relu(self.bottleneck(x))
        x = torch.relu(self.dec1(x))
        x = self.up1(x)
        x = torch.relu(self.dec2(x))
        x = self.up2(x)
        return torch.sigmoid(self.out(x))
