import numpy as np
import torch


def patch_initialization(image_size=(3, 224, 224), noise_percentage=0.5):
    # if patch_type == 'rectangle':  # 矩形补丁
    mask_length = int((noise_percentage * image_size[1] * image_size[2]) ** 0.5)
    patch = np.random.rand(image_size[0], mask_length, mask_length)
    return patch


def clamp_patch(patch, dset):  # 调整patch的数值上下限
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    min_in = np.array([0, 0, 0])
    max_in = np.array([1, 1, 1])
    min_out, max_out = np.min((min_in - mean) / std), np.max((max_in - mean) / std)  # 正则化min_in 和 max_in
    patch = torch.clamp(patch, min=min_out, max=max_out)
    return patch


def un_normalize(x, dset):
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    x[0] = x[0] * std[0] + mean[0]
    x[1] = x[1] * std[1] + mean[1]
    x[2] = x[2].mul(std[2]) + mean[2]
    return x

