import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from functools import lru_cache

import torch
from torch.utils import data
from torchvision import datasets


class OmniMNIST(data.Dataset):
    def __init__(self, root='data/datas/MNIST', train=True, download=True,
                 fov=90, outshape=(60, 60),
                 h_rotate=True, v_rotate=True, fix_aug=False):

        self.dataset = datasets.MNIST(root, train=train, download=download)
        self.fov = fov
        self.outshape = outshape
        self.h_rotate = h_rotate
        self.v_rotate = v_rotate

        self.aug = None
        # fix augmentation for testing data
        if fix_aug:
            self.aug = [
                {
                    'h_rotate': np.random.randint(outshape[1]),
                    'v_rotate': np.random.uniform(-np.pi/2, np.pi/2),
                }
                for _ in range(len(self.dataset))
            ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = np.array(self.dataset[idx][0], np.float32)
        h, w = img.shape[:2]
        uv = genuv(*self.outshape)
        fov = self.fov * np.pi / 180

        if self.v_rotate:
            if self.aug is not None:
                v_c = self.aug[idx]['v_rotate']
            else:
                v_c = np.random.uniform(-np.pi/2, np.pi/2)
            img_idx = uv2img_idx(uv, h, w, fov, fov, v_c)
        else:
            img_idx = uv2img_idx(uv, h, w, fov, fov, 0)
        x = map_coordinates(img, img_idx, order=1)

        # Random horizontal rotate
        if self.h_rotate:
            if self.aug is not None:
                dx = self.aug[idx]['h_rotate']
            else:
                dx = np.random.randint(x.shape[1])
            x = np.roll(x, dx, axis=1)
            

        return torch.FloatTensor(x.copy()), self.dataset[idx][1]


def genuv(h, w):
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = (u + 0.5) * 2 * np.pi / w - np.pi
    v = (v + 0.5) * np.pi / h - np.pi / 2
    return np.stack([u, v], axis=-1)


def uv2xyz(uv):
    sin_u = np.sin(uv[..., 0])
    cos_u = np.cos(uv[..., 0])
    sin_v = np.sin(uv[..., 1])
    cos_v = np.cos(uv[..., 1])
    return np.stack([
        cos_v * cos_u,
        cos_v * sin_u,
        sin_v
    ], axis=-1)


def xyz2uv(xyz):
    c = np.sqrt((xyz[..., :2] ** 2).sum(-1))
    u = np.arctan2(xyz[..., 1], xyz[..., 0])
    v = np.arctan2(xyz[..., 2], c)
    return np.stack([u, v], axis=-1)


def uv2img_idx(uv, h, w, u_fov, v_fov, v_c=0):
    assert 0 < u_fov and u_fov < np.pi
    assert 0 < v_fov and v_fov < np.pi
    assert -np.pi < v_c and v_c < np.pi

    xyz = uv2xyz(uv.astype(np.float64))
    Ry = np.array([
        [np.cos(v_c), 0, -np.sin(v_c)],
        [0, 1, 0],
        [np.sin(v_c), 0, np.cos(v_c)],
    ])
    xyz_rot = xyz.copy()
    xyz_rot[..., 0] = np.cos(v_c) * xyz[..., 0] - np.sin(v_c) * xyz[..., 2]
    xyz_rot[..., 1] = xyz[..., 1]
    xyz_rot[..., 2] = np.sin(v_c) * xyz[..., 0] + np.cos(v_c) * xyz[..., 2]
    uv_rot = xyz2uv(xyz_rot)

    u = uv_rot[..., 0]
    v = uv_rot[..., 1]

    x = np.tan(u)
    y = np.tan(v) / np.cos(u)
    x = x * w / (2 * np.tan(u_fov / 2)) + w / 2
    y = y * h / (2 * np.tan(v_fov / 2)) + h / 2

    invalid = (u < -u_fov / 2) | (u > u_fov / 2) |\
              (v < -v_fov / 2) | (v > v_fov / 2)
    x[invalid] = -100
    y[invalid] = -100

    return np.stack([y, x], axis=0)


