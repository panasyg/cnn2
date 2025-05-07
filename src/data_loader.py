import os
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils.preprocessing import (
    load_volume, normalize, apply_clahe,
    random_flip_3d, random_noise, random_crop_3d
)

class LandmineDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        patch_size: tuple = (64,64,64),
        mode: str = 'train'
    ):
        """
        mode: 'train' | 'val' | 'test'
        на train — застосовуємо аугментації, на val/test — лише детерміністичні препроцеси
        """
        self.files = [
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith('.npy')
        ]
        self.patch_size = patch_size
        self.mode = mode

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        raw = np.load(self.files[idx], allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.shape == ():
            raw = raw.item()
        vol = raw['data'] if isinstance(raw, dict) and 'data' in raw else raw

        # нормалізація + CLAHE
        vol = normalize(vol)
        vol = apply_clahe(vol)

        # на train — crop + шум + flip
        if self.mode == 'train':
            vol = random_crop_3d(vol, self.patch_size)
            vol = random_flip_3d(vol, p=0.5)
            vol = random_noise(vol, std=0.01)
        else:
            # на val/test — центроване обрізання
            D, H, W = vol.shape
            z0 = (D - self.patch_size[0]) // 2
            y0 = (H - self.patch_size[1]) // 2
            x0 = (W - self.patch_size[2]) // 2
            vol = vol[z0:z0+self.patch_size[0],
                      y0:y0+self.patch_size[1],
                      x0:x0+self.patch_size[2]]

        # у форму для Conv3d
        vol = np.expand_dims(vol, axis=0).astype(np.float32)  # (1, D, H, W)
        return torch.from_numpy(vol)
