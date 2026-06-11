import torch
import numpy as np
import random


# 训练图像的label都属于哪几类: index -> hashcenters[index]
def get_org_index(list_path):
    lines = open(list_path).readlines()
    num = len(lines)
    index = []
    for i in range(num):
        label = lines[i].replace('\n', '').split(' ')[1:]
        index_ = label.index('1')
        index.append(index_)
    return set(index)


def voting_center(list_path, hashcenters, hash_bit):  # 投票得到centeOfcenter
    index = get_org_index(list_path)
    pos = [0 for i in range(hash_bit)]
    neg = [0 for i in range(hash_bit)]
    for ind in index:
        center = torch.from_numpy(hashcenters[ind]).unsqueeze(0)
        for bit in range(hash_bit):  # 统计每个hashcenter的每一bit是pos/neg
            if center[0][bit] == 1:
                pos[bit] += 1
            else:  # -1
                neg[bit] += 1
    target = [0 for i in range(hash_bit)]  # centerOfcenter  64bit的-1 / 1
    for bit in range(hash_bit):
        if pos[bit] >= neg[bit]:
            target[bit] = 1
        else:
            target[bit] = -1
    target = torch.tensor(target).to(torch.float32)
    return target


def compute_mean_center(list_path, hashcenters, hash_bit):  # 取平均得到centeOfcenter
    index = list(get_org_index(list_path))
    centers = torch.from_numpy(hashcenters[index[0]]).unsqueeze(0)  # 初始化(第一个hashcenter)
    for i in range(len(index) - 1):
        center_ = torch.from_numpy(hashcenters[index[i + 1]]).unsqueeze(0)  # 从第2个hashcenter开始
        centers = torch.cat([centers, center_])
    center = centers.mean(dim=0)  # 每位取平均
    return center


def voting(inds, hashcenters, hash_bit, is_father=False):
    pos = [0 for _ in range(hash_bit)]
    neg = [0 for _ in range(hash_bit)]
    for ind in inds:
        if is_father:
            class_rand = 1  # torch.rand(1)
        else:
            class_rand = 1
        center = torch.from_numpy(hashcenters[ind]).unsqueeze(0)
        for bit in range(hash_bit):
            if center[0][bit] == 1:
                pos[bit] += 1 * class_rand
            else:  # -1
                neg[bit] += 1 * class_rand
    target = [0 for _ in range(hash_bit)]
    for bit in range(hash_bit):
        if pos[bit] >= neg[bit]:
            target[bit] = 1
        else:
            target[bit] = -1
    target = torch.tensor(target).to(torch.float32)
    return target


def voting_anchors(hashcenters, num_spts=1, hash_bit=64, is_father=False, min_idx=3):
    max_anchors = len(hashcenters)  # 28类
    if is_father:
        inds = [i for i in range(max_anchors)]  # (全类别的总负锚点)
        anchor = voting(inds, hashcenters, hash_bit, is_father)
        return anchor
    else:
        anchor_sets = []
        for j in range(num_spts):  # 随机聚集若干个类，得到一系列子锚点
            rand_num_of_classes = np.random.randint(2, max_anchors)
            inds = random.sample(list(range(min_idx, max_anchors)), min(rand_num_of_classes, max_anchors - min_idx))

            sub_anchor = voting(inds, hashcenters, hash_bit)
            anchor_sets.append(sub_anchor)
        sub_anchors = np.stack(anchor_sets)
        return sub_anchors


def voting_anchors_oldest(hashcenters, num_spts, hash_bit, is_father=False, min_idx=3):
    max_anchors = len(hashcenters)  # 28类
    if is_father:
        inds = [i for i in range(max_anchors)]  # (全类别的总负锚点)
        anchor = voting(inds, hashcenters, hash_bit, is_father)
        return anchor
    else:
        anchor_sets = []
        for j in range(num_spts):  # 随机聚集若干个类，得到一系列子锚点
            rand_num_of_classes = np.random.randint(3, max_anchors - 1 - 3)
            inds = random.sample(range(3, max_anchors - 1), rand_num_of_classes)
            sub_anchor = voting(inds, hashcenters, hash_bit)
            anchor_sets.append(sub_anchor)
        sub_anchors = np.stack(anchor_sets)
        return sub_anchors


def voting_new(inds, hashcenters, hash_bit):  # 投票正负1
    hashcenters_ = hashcenters[inds]
    hashcenters_ = np.sum(hashcenters_, axis=0)
    hashcenters_[hashcenters_ > 0] = 1
    hashcenters_[hashcenters_ < 0] = -1
    zero_ind = np.where(hashcenters_ == 0)[0]
    random_values = np.random.choice([-1, 1], size=zero_ind.shape)
    hashcenters_[zero_ind] = random_values
    hashcenters_[hashcenters_ == 0] = np.random.choice([-1, 1])
    return torch.tensor(hashcenters_).to(torch.float32)


def voting_anchors_new(hashcenters, num_spts, hash_bit, is_father=False, min_idx=3):
    max_anchors = len(hashcenters)  # 28类
    if is_father:
        inds = [i for i in range(max_anchors)]  # (全类别的总负锚点)
        anchor = voting_new(inds, hashcenters, hash_bit)
        return anchor
    else:
        anchor_sets = []
        for j in range(num_spts):  # 随机聚集若干个类，得到一系列子锚点
            rand_num_of_classes = np.random.randint(2, max_anchors)
            inds = random.sample(list(range(min_idx, max_anchors)), min(rand_num_of_classes, max_anchors - min_idx))
            sub_anchor = voting_new(inds, hashcenters, hash_bit)
            anchor_sets.append(sub_anchor)
        sub_anchors = np.stack(anchor_sets)
        return sub_anchors


def voting_center_(list_path, hashcenters, hash_bit):  # 为每个source_txt的图, 根据label得到hashcenter
    lines = open(list_path).readlines()
    num = len(lines)
    index = []
    for i in range(num):
        label = lines[i].replace('\n', '').split(' ')[1:]
        index_ = label.index('1')
        index.append(index_)
    target = []
    for i in range(num):
        target.append(hashcenters[index[i]])

    target = torch.tensor(target).to(torch.float32)
    return target
