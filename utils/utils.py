import os
import numpy as np
import torch
import random
from torch.autograd import Variable
from tqdm import tqdm
from collections import defaultdict
import datetime
import pandas as pd
import torch.nn as nn 
import torch.nn.functional as F 
import sys
import copy
import math  

def random_jit(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data = np.add(jittered_data, batch_data)
    return jittered_data

def random_sca(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data
    
def random_rot(batch_data):
    """ Randomly rotate the point clouds"""
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32) 

    x = np.random.uniform() * 2 * np.pi
    y = np.random.uniform() * 2 * np.pi
    z = np.random.uniform() * 2 * np.pi

    x_matrix = np.array([[1, 0, 0], [0, np.cos(x), np.sin(x)], [0, -np.sin(x), np.cos(x)]])
    y_matrix = np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])
    z_matrix = np.array([[np.cos(z), np.sin(z), 0], [-np.sin(z), np.cos(z), 0], [0, 0, 1]])
    rotation_matrix = np.dot(x_matrix, y_matrix)
    rotation_matrix = np.dot(rotation_matrix, z_matrix)

    for k in range(batch_data.shape[0]): 
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


class SORDefense(nn.Module):
    """Statistical outlier removal as defense"""

    def __init__(self, k=2, alpha=1.1, npoint=1024): 
        super(SORDefense, self).__init__()
        self.k = k
        self.alpha = alpha
        self.npoint = npoint

    def outlier_removal(self, x):
        """Removes large kNN distance points"""
        pc = x.clone().detach().double()
        B, K = pc.shape[:2]
        pc = pc.transpose(2, 1)  # [B, 3, K]
        inner = -2. * torch.matmul(pc.transpose(2, 1), pc)  # [B, K, K]
        xx = torch.sum(pc ** 2, dim=1, keepdim=True)  # [B, 1, K]
        dist = xx + inner + xx.transpose(2, 1)  # [B, K, K]
        assert dist.min().item() >= -1e-6
        # the min is self so we take top (k + 1)
        neg_value, _ = (-dist).topk(k=self.k + 1, dim=-1)  # [B, K, k + 1]
        value = -(neg_value[..., 1:])  # [B, K, k]
        value = torch.mean(value, dim=-1)  # [B, K]
        mean = torch.mean(value, dim=-1)  # [B]
        std = torch.std(value, dim=-1)  # [B]
        threshold = mean + self.alpha * std  # [B]
        bool_mask = (value <= threshold[:, None])  # [B, K]
        sel_pc = x[0][bool_mask[0]].unsqueeze(0)
        sel_pc = self.process_data(sel_pc)
        for i in range(1, B):
            proc_pc = x[i][bool_mask[i]].unsqueeze(0)
            proc_pc = self.process_data(proc_pc)
            sel_pc = torch.cat([sel_pc, proc_pc], dim=0)
        return sel_pc

    def process_data(self, pc, npoint=None): 
        if npoint is None:
            npoint = self.npoint
        proc_pc = pc.clone()
        num = npoint // pc.size(1)
        for _ in range(num-1):
            proc_pc = torch.cat([proc_pc, pc], dim=1)
        num = npoint - proc_pc.size(1)
        duplicated_pc = proc_pc[:, :num, :]
        proc_pc = torch.cat([proc_pc, duplicated_pc], dim=1)
        assert proc_pc.size(1) == npoint
        return proc_pc

    def forward(self, x):
        with torch.enable_grad():
            x = self.outlier_removal(x)
            x = self.process_data(x)  # to batch input
        return x


def show_time(time_now):
    past_day = int(time_now / 60 / 60 / 24)  # past days from 1970
    hour = int((time_now - past_day * 24 * 60 * 60) / 60 / 60)  # past hours
    start_minute = int((time_now - past_day * 24 * 60 * 60 - hour * 60 * 60) / 60)  # past minutes
    start_hour = int(hour + 8)  # GMT + 8
    if (start_hour >= 24):  
        start_hour = start_hour - 24 
    return start_hour, start_minute

def transform_time(start_time, end_time):
    start_hour, start_minute = show_time(start_time)
    end_hour, end_minute = show_time(end_time)

    if end_hour < start_hour:  
        spent_hour = 24 - start_hour + end_hour
    else:
        spent_hour = end_hour - start_hour
    if end_minute < start_minute:
        spent_hour -= 1
        spent_minute = end_minute + 60 - start_minute
    else:
        spent_minute = end_minute - start_minute
    print("Start time (GMT+8) is {}:{}".format(start_hour, start_minute))
    print("End time (GMT+8) is {}:{}".format(end_hour, end_minute))
    return spent_hour, spent_minute
  

def set_seed(seed = 0):
    print('Using random seed', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class PointNet(nn.Module):
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x


class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x