import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader.ModelNetDataLoader40 import ModelNetDataLoader40
from dataloader.ModelNetDataLoader10 import ModelNetDataLoader10
from dataloader.ShapeNetDataLoader import PartNormalDataset
from torch.utils.data import DataLoader, TensorDataset

from utils.logging import Logging_str 
from utils.utils import show_time, transform_time, set_seed 
import math
import os
import sys
import time
import random
import importlib
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'classifiers'))

 
 

def load_data(args, data_path):
    """Load the dataset from the given path"""
    print('Start Loading Dataset...')
    if args.dataset == 'ModelNet40':
        DATASET = ModelNetDataLoader40(
            root=data_path,
            npoint=args.input_point_nums,
            split='train',
            normal_channel=False
        )

    elif args.dataset == 'ModelNet10':
        DATASET = ModelNetDataLoader10(
            root=data_path,
            npoint=args.input_point_nums,
            split='train',
            normal_channel=False
        )


    elif args.dataset == 'ShapeNetPart':
        DATASET = PartNormalDataset(
            root=data_path,
            npoint=args.input_point_nums,
            split='train',
            normal_channel=False
        )
    else:
        raise NotImplementedError

    T_DataLoader = torch.utils.data.DataLoader(
        DATASET,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print('Finish Loading Dataset...')
    return T_DataLoader

def data_preprocess(data):
    """Preprocess the given data and label"""
    points, target = data

    points = points # [B, N, C]
    target = target[:, 0] # [B]

    points = points.cuda()
    target = target.cuda()

    return points, target

def save_tensor_as_txt(args, points, filename):  
    """Save the torch tensor into a txt file"""
    points = points.squeeze(0).detach().cpu().numpy()
    file_path = os.path.join(args.output_dir)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(os.path.join(file_path,filename), "w") as f:
        for i in range(points.shape[0]):
            msg = str(points[i][0]) + ' ' + str(points[i][1]) + ' ' + str(points[i][2]) 
            f.write(msg+'\n')
        f.close()

def rotation_pc(pointcloud, del_1, del_2, del_3):
    pointcloud = pointcloud.clone().detach().cpu().numpy()
    alpha = np.pi / 180. * del_1
    gamma = np.pi / 180. * del_2
    beta = np.pi / 180. * del_3

    matrix_1 = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    matrix_2 = np.array([[np.cos(gamma), 0, np.sin(gamma)], [0, 1, 0], [-np.sin(gamma), 0, np.cos(gamma)]])
    matrix_3 = np.array([[np.cos(beta), -np.sin(beta), 0], [np.sin(beta), np.cos(beta), 0], [0, 0, 1]])

    new_pc = np.matmul(pointcloud, matrix_1)
    new_pc = np.matmul(new_pc, matrix_2)
    new_pc = np.matmul(new_pc, matrix_3).astype('float32')
    return new_pc

def output_delta_list(del_13_list, del_2_list, num_class):
    if len(del_13_list) * len(del_2_list) < num_class:
        print("Excessive interval! ")
        exit(-1)
    all_combine_list, del_list = [], []
    for i in range(len(del_13_list)):
        for j in range(len(del_2_list)):
            del_1 = del_13_list[i][0]
            del_2 = del_2_list[j]
            del_3 = del_13_list[i][1]
            all_combine_list.append([del_1, del_2, del_3])
    del_list = random.sample(all_combine_list, num_class)
    print(del_list)
    return del_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointAPA: Towards Availability Poisoning Attacks in 3D Point Clouds')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N', help='input batch size for training (default: 1)')
    parser.add_argument('--input_point_nums', type=int, default=1024, help='Point nums of each point cloud')
    parser.add_argument('--seed', type=int, default=2022, metavar='S', help='random seed (default: 2022)')
    parser.add_argument('--dataset', type=str, default='ModelNet10', choices=['ModelNet10', 'ModelNet40', 'ShapeNetPart'])
    parser.add_argument('--num_workers', type=int, default=4, help='Worker nums of data loading.')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]') 
    parser.add_argument('--interval', type=int, default=42, help='The interval of rotation')   #ablation study of interval angle
 

    args = parser.parse_args()
    args.device = torch.device("cuda")
    set_seed(args.seed)
    num_class = 0
    if args.dataset == 'ModelNet40':
        num_class = 40
        data_path = "./clean_data/modelnet40_normal_resampled"

    elif args.dataset == 'ShapeNetPart':
        num_class = 16
        data_path = './clean_data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'

    elif args.dataset == 'ModelNet10':
        num_class = 10
        data_path = "./clean_data/modelnet40_normal_resampled"

    assert num_class != 0

    args.num_class = num_class
    args.output_dir = os.path.join("poison_data", args.dataset, str(args.interval))
    del_13_list = [[0,0], [0,10], [0,20], [10,10], [10,20], [20,20]]        #alpha and beta angle list
    del_2_list = []     #gamma angle list
    for i in range(math.ceil(360/args.interval)-1):
        del_2_list.append(args.interval*(i+1))
    delta_list = output_delta_list(del_13_list, del_2_list, num_class)



    dataloader = load_data(args, data_path)
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    start = time.time() 

    for batch_id, data in pbar:
        if args.dataset == 'ShapeNetPart':
            data = data[:2]
        points, target = data_preprocess(data)
        target = target.long()
 
        """PointAPA poisoning process"""
        poi_points = rotation_pc(points, del_1=delta_list[target.item()][0], del_2=delta_list[target.item()][1], del_3=delta_list[target.item()][2])
        poi_points = torch.tensor(poi_points).cuda()
        #del_2 controls z axis

        save_tensor_as_txt(args, poi_points, f'{batch_id}_poi_{target.item()}.txt') 

        """visualizing point cloud samples"""
        # from utils.visual_util import plot_pcd_three_views
        # titles = ['viewpoint 1', 'viewpoint 2', 'viewpoint 3']
        # file_path = os.path.join(args.output_dir, 'fig')
        # if not os.path.exists(file_path):
        #     os.makedirs(file_path)
        # plot_pcd_three_views(os.path.join(file_path, f'{batch_id}_ori_{target.item()}.png'),[points.squeeze(0).detach().cpu().numpy()],titles)
        # plot_pcd_three_views(os.path.join(file_path,f'{batch_id}_poi_{target.item()}.png'),[poi_points.squeeze(0).detach().cpu().numpy()], titles)
    end = time.time() 
    spent_hour, spent_min = transform_time(start, end)
    print("The time overhead of PointAPA is {} h {} min".format(spent_hour, spent_min))
 