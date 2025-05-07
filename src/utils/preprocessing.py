import numpy as np
import cv2

def load_volume(path: str) -> np.ndarray:
    """Завантажує .npy (можливо з pickle-об’єктом) і повертає чистий 3D-масив."""
    raw = np.load(path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, dict) and 'data' in raw:
        return raw['data']
    return raw

def normalize(volume: np.ndarray) -> np.ndarray:
    """Zero–one нормалізація по всьому об’єму."""
    min_v, max_v = volume.min(), volume.max()
    return (volume - min_v) / (max_v - min_v + 1e-8)

def apply_clahe(volume: np.ndarray) -> np.ndarray:
    """CLAHE по кожному 2D-зрізу та повторна нормалізація."""
    d, h, w = volume.shape
    processed = np.zeros_like(volume)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(d):
        slice_8bit = (volume[i] * 255).astype(np.uint8)
        dst = clahe.apply(slice_8bit)
        p = dst.astype(np.float32) / 255.0
        processed[i] = (p - p.min()) / (p.max() - p.min() + 1e-8)
    return processed

def random_flip_3d(volume: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Випадкове дзеркальне відображення по кожній осі з ймовірністю p."""
    if np.random.rand() < p:
        if np.random.rand() < 0.5:
            volume = volume[::-1, ...]
        if np.random.rand() < 0.5:
            volume = volume[:, ::-1, ...]
        if np.random.rand() < 0.5:
            volume = volume[:, :, ::-1]
    return volume

def random_noise(volume: np.ndarray, std: float = 0.01) -> np.ndarray:
    """Додавання гаусового шуму."""
    noise = np.random.normal(0, std, size=volume.shape)
    return np.clip(volume + noise, 0.0, 1.0)

def random_crop_3d(volume: np.ndarray, output_size: tuple) -> np.ndarray:
    """Рандомне обрізання до output_size (ps_d, ps_h, ps_w)."""
    D, H, W = volume.shape
    ps_d, ps_h, ps_w = output_size
    assert D >= ps_d and H >= ps_h and W >= ps_w, "Patch size > volume size"
    z = np.random.randint(0, D - ps_d + 1)
    y = np.random.randint(0, H - ps_h + 1)
    x = np.random.randint(0, W - ps_w + 1)
    return volume[z:z+ps_d, y:y+ps_h, x:x+ps_w]
