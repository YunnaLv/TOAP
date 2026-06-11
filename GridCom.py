import math
import random

import numpy as np
import torch.nn.functional as F
import torch
from torchvision import transforms


class GridCom:
    def __init__(self, com_size, ComG):
        super().__init__()
        self.com_size = com_size
        self.ComG = ComG

    def GridAndCom(self, img, grid_size=(112, 112), step=(56, 56)):
        # batch_size个图像，3x224x224分割成9个3x112x112的网格，步长是56
        # （就是说图像之间会有重合的地方）
        batch_size, channels, height, width = img.size()
        grids_list = []

        for img_ in img:
            for y in range(0, height - grid_size[0] + 1, step[0]):
                for x in range(0, width - grid_size[1] + 1, step[1]):
                    grid = img_[:, y: y + grid_size[0], x: x + grid_size[1]]
                    grids_list.append(grid)
        grids = torch.stack(grids_list)
        grids = grids.contiguous().view(-1, channels, grid_size[0], grid_size[1])
        com_grids = self.ComG(grids, h=112, w=112)
        com_grids = com_grids.contiguous().view(batch_size, -1, channels, grid_size[0], grid_size[1])
        return com_grids

    def merge_imgs(self, grids, output_size=(224, 224), step=56):
        batch_size, num_grids, channels, h, w = grids.size()

        output_height, output_width = output_size

        # 初始化合并后的张量和计数张量
        output_tensor = torch.zeros(batch_size, channels, output_height, output_width)
        count_tensor = torch.zeros(batch_size, channels, output_height, output_width)

        # 对每张图像进行滑动合并
        for i in range(num_grids):
            grids = grids.cpu()
            y_offset = (i // 3) * step  # 计算 y 方向的偏移
            x_offset = (i % 3) * step  # 计算 x 方向的偏移

            output_tensor[:, :, y_offset:y_offset + h, x_offset:x_offset + w] += grids[:, i, :, :, :]
            count_tensor[:, :, y_offset:y_offset + h, x_offset:x_offset + w] += 1
        # 计算最终的平均值
        output_tensor = output_tensor / count_tensor

        return output_tensor


class GridComDropout:
    def __init__(self, com_size, ComG):
        super().__init__()
        self.com_size = com_size
        self.ComG = ComG

    def GridAndCom(self, img, grid_size=(112, 112), step=(56, 56)):
        # batch_size个图像，3x224x224分割成9个3x112x112的网格，步长是56
        # （就是说图像之间会有重合的地方）
        batch_size, channels, height, width = img.size()
        grids_list = []

        for img_ in img:
            for y in range(0, height - grid_size[0] + 1, step[0]):
                for x in range(0, width - grid_size[1] + 1, step[1]):
                    grid = img_[:, y: y + grid_size[0], x: x + grid_size[1]]
                    grids_list.append(grid)
        grids = torch.stack(grids_list)
        grids = grids.contiguous().view(-1, channels, grid_size[0], grid_size[1])
        com_grids = self.ComG(grids, h=112, w=112)
        com_grids = com_grids.contiguous().view(batch_size, -1, channels, grid_size[0], grid_size[1])
        return com_grids

    def merge_imgs(self, grids, output_size=(224, 224), step=56, dropnum=2, grid_list=None):
        batch_size, num_grids, channels, h, w = grids.size()

        output_height, output_width = output_size

        # 初始化合并后的张量和计数张量
        output_tensor = torch.zeros(batch_size, channels, output_height, output_width)
        count_tensor = torch.zeros(batch_size, channels, output_height, output_width)

        # 对每张图像进行滑动合并
        if grid_list == None:
            grid_list = [0, 2, 6, 8]
            grid_list += random.sample([1, 3, 4, 5, 7], 5 - dropnum)
        # for i in range(num_grids):
        for i in grid_list:
            grids = grids.cpu()
            y_offset = (i // 3) * step  # 计算 y 方向的偏移
            x_offset = (i % 3) * step  # 计算 x 方向的偏移

            output_tensor[:, :, y_offset:y_offset + h, x_offset:x_offset + w] += grids[:, i, :, :, :]
            count_tensor[:, :, y_offset:y_offset + h, x_offset:x_offset + w] += 1
        # 计算最终的平均值
        output_tensor = output_tensor / count_tensor

        return output_tensor

