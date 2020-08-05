import numpy as np
from numpy import pi, sin, cos, tan, arcsin, arctan
from functools import lru_cache

import torch
from torch import nn
import  torch.nn.functional as F
from torch.nn.parameter import Parameter


class SphereConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super(SphereConv, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.weight = Parameter(torch.Tensor(out_c, in_c, 3, 3))
        self.bias = Parameter(torch.Tensor(out_c))
        # weights init
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        self.bias.data.zero_()
    
    def forward(self, x):
        # get sampling pattern
        pattern = kernel_sampling_pattern(x.shape[2], x.shape[3], self.stride)
        pattern = torch.FloatTensor(pattern).to(x.device)

        grid = pattern.repeat(x.shape[0], 1, 1, 1)
        x = F.grid_sample(x, grid)
        x = F.conv2d(x, self.weight, self.bias, stride=3)
        return x


class SpherePool(nn.Module):
    def __init__(self, stride=1):
        super(SpherePool, self).__init__()
        self.stride = stride
    
    def forward(self, x):
        pattern = kernel_sampling_pattern(x.shape[2], x.shape[3], self.stride)
        pattern = torch.FloatTensor(pattern).to(x.device)

        grid = pattern.repeat(x.shape[0], 1, 1, 1)
        x = F.grid_sample(x, grid)
        return F.max_pool2d(x, kernel_size=3)
    
    
@lru_cache(None)
def kernel_sampling_pattern(h, w, stride):
    # pattern shape: (W_in, H_in, 3, 3, 2)
    # torch.nn.functional.grid_sample require grid with shape: (N, H_out ,W_out ,2)
    pattern = np.array([[get_absolute_positions(h, w, x, y) 
                         for y in range(0, h, stride)] 
                   for x in range(0, w, stride)])
    # grid location should be normalized in (-1, 1)
    pattern[..., 0] = (pattern[..., 0] * 2 / w) - 1
    pattern[..., 1] = (pattern[..., 1] * 2 / h) - 1
    pattern = pattern.transpose(0, 2, 1, 3, 4)
    s = pattern.shape
    pattern = pattern.reshape(1, s[0]*s[1], s[2]*s[3], 2)
    return pattern.copy()


@lru_cache(None)
def get_absolute_positions(h, w, center_x, center_y):
    relative_positions = get_relative_positions(w, h)  # 9 points' coordinates
    x = relative_positions[..., 0]
    y = relative_positions[..., 1]
    rho = np.sqrt(x**2+y**2)
    v = arctan(rho)
    offset = 0.1  # avoid division error

    # pixel to radian
    center_theta = (center_x + offset) * 2 * pi / w  - pi
    center_phi = pi / 2 - (center_y + offset) * pi / h

    # inverse gnomonic projection
    phi= arcsin(cos(v) * sin(center_phi) + y * sin(v) * cos(center_phi) / rho)
    theta = center_theta + arctan(x * sin(v) / (rho * cos(center_phi) * cos(v) - y * sin(center_phi) * sin(v)))
    # img_x, img_y: coordinates on equirectangular image
    img_x = ((theta + pi) * w / pi / 2 - offset) % w  # cross equirectangular image boundary 
    img_y = (pi / 2 - phi) * h / pi - offset

    # todo
    absolute_positions = np.stack([img_x, img_y], axis=-1)
    absolute_positions[1, 1] = (center_x, center_y)
    return absolute_positions



@lru_cache(None)
def get_relative_positions(w, h):
    # delta phi and delta theta: the step size on sphere
    d_phi = pi/h
    d_theta = 2*pi/w
    # the relative positions to center point
    return np.array([
        [
            [-tan(d_theta), 1/cos(d_theta)*tan(d_phi)], 
            [0,             tan(d_phi)], 
            [tan(d_theta),  1/cos(d_theta)*tan(d_phi)]
        ],
        [
            [-tan(d_theta), 0], 
            [0.5,         0.5], 
            [tan(d_theta),  0]
        ],
        [
            [-tan(d_theta), -1/cos(d_theta)*tan(d_phi)], 
            [0,             -tan(d_phi)], 
            [tan(d_theta),  -1/cos(d_theta)*tan(d_phi)]
        ]
    ])


