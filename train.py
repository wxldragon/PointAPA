import os,sys,argparse,time,torch 
import sklearn.metrics as metrics
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import importlib
import torch.optim as optim
from tqdm import tqdm 
from torch.utils.data import DataLoader, TensorDataset 
from utils.logging import Logging_str
from utils.utils import set_seed,random_jit,random_sca,random_rot, SORDefense, show_time, transform_time, PointNet, DGCNN
import math, random
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataloader.ModelNetDataLoader40 import ModelNetDataLoader40
from dataloader.ModelNetDataLoader10 import ModelNetDataLoader10
from dataloader.ShapeNetDataLoader import PartNormalDataset 


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'classifiers'))

def load_poison_train_data(args, data_path):
    file_names = os.listdir(data_path) 
    assert len(file_names) > 0, 'No poisoned data found! Run poison_generation.py to obtain poisoned data!'
    file_names = sorted(file_names)
    dataset, labels = [], []
    for fn in tqdm(file_names):
        file_path = os.path.join(data_path, fn)
        pc = np.loadtxt(file_path).astype(np.float32)
        dataset.append(pc)
        labels.append(fn.split('.')[0].split('_')[-1])

    dataset = torch.from_numpy(np.array(dataset))
    labels = torch.from_numpy(np.array(labels).astype(np.float32)).unsqueeze(1)
    DATASET = TensorDataset(dataset, labels)
    dataloader = DataLoader(DATASET, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=True) 
    return dataloader


def load_clean_train_data(args, data_path):
    if args.dataset == 'ModelNet40':
        DATASET = ModelNetDataLoader40(root=data_path, npoint=args.input_point_nums, split='train', normal_channel=False)
    elif args.dataset == 'ModelNet10':
        DATASET = ModelNetDataLoader10(root=data_path, npoint=args.input_point_nums, split='train', normal_channel=False)
    elif args.dataset == 'ShapeNetPart':
        DATASET = PartNormalDataset(root=data_path, npoint=args.input_point_nums, split='train', normal_channel=False) 
    else:
        raise NotImplementedError

    T_DataLoader = torch.utils.data.DataLoader(DATASET, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=True)
    return T_DataLoader



def data_preprocess(data):
    """Preprocess the given data and label"""
    points, target = data
    points = points # [B, N, C]
    target = target[:, 0] # [B]
    points = points.cuda()
    target = target.cuda().long()
    return points, target

def build_models(args):
    MODEL = importlib.import_module(args.target_model)
    classifier = MODEL.get_model(args.NUM_CLASSES, normal_channel=False)
    classifier = classifier.to(args.device)
    return classifier


def cal_loss(pred, gold): 
    gold = gold.contiguous().view(-1)    
    loss = F.cross_entropy(pred, gold, reduction='mean')
    return loss


def main():
    if args.target_model == 'pointnet_cls':
        model = PointNet(args, output_channels=args.NUM_CLASSES).cuda()
    elif args.target_model == 'dgcnn':
        model = DGCNN(args, output_channels=args.NUM_CLASSES).cuda()
    else:
        model = build_models(args).cuda() 


    """loading clean test data"""
    if args.dataset == 'ModelNet40':
        args.NUM_CLASSES = 40
        clean_path = "clean_data/modelnet40_normal_resampled"
        test_dataset = ModelNetDataLoader40(root=clean_path, npoint=args.input_point_nums, split='test', normal_channel=False)
    elif args.dataset == 'ModelNet10':
        args.NUM_CLASSES = 10
        clean_path = "clean_data/modelnet40_normal_resampled"
        test_dataset = ModelNetDataLoader10(root=clean_path, npoint=args.input_point_nums, split='test', normal_channel=False)
    elif args.dataset == 'ShapeNetPart':
        args.NUM_CLASSES = 16
        clean_path = "clean_data/shapenetcore_partanno_segmentation_benchmark_v0_normal"
        test_dataset = PartNormalDataset(root=clean_path, npoint=args.input_point_nums, split='test', normal_channel=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,drop_last=True)



    if args.poison_train:   #loading PointAPA training data
        poison_path = os.path.join("./poison_data", args.dataset, str(args.interval))
        train_loader = load_poison_train_data(args, poison_path)
        print("Start poison training (interval angle = {}) using model {}".format(args.interval, args.target_model))
    else:       #loading clean training data
        train_loader = load_clean_train_data(args, clean_path)
        print("Start clean training using model {}".format(args.target_model))

 
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epoch, eta_min=args.lr)
    criterion = cal_loss 
    test_acc_list, train_acc_list = [], []
    for epoch in range(args.epoch):
        scheduler.step()
        train_loss, count = 0.0, 0.0
        model.train()
        train_pred, train_true = [], []

        for data in tqdm(train_loader):
            if args.dataset == 'ShapeNetPart':
                data = data[:2]
            data, label = data_preprocess(data)
            
            """Employing defense schemes"""
            if args.defense: 
                if args.aug_type == 'sca':
                    data = data.clone().detach().cpu().numpy()
                    data = random_sca(data)
                elif args.aug_type == 'rot':
                    data = data.clone().detach().cpu().numpy()
                    data = random_rot(data)
                elif args.aug_type == 'jit':
                    data = data.clone().detach().cpu().numpy()
                    data = random_jit(data)
                elif args.aug_type == 'sor':
                    sor = SORDefense(k=2, alpha=1.1)
                    data = sor(data)
                else:
                    print("Wrong defense type!")
                    exit(-1)
                data = torch.tensor(data).to(torch.float32)


            data, label = data.cuda(), label.long().cuda().squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        round_acc = round(train_acc*100, 2)
        train_acc_list.append(round_acc)
        print('Epoch[%d/%d] loss: %.4f, train acc: %.4f' % (epoch + 1, args.epoch,  train_loss * 1.0 / count, train_acc))

        test_loss, count = 0.0, 0.0
        model.eval()
        test_pred,test_true = [], []

        for data in tqdm(test_loader):
            if args.dataset == 'ShapeNetPart':
                data = data[:2] 
            data, label = data_preprocess(data)

            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())

        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        round_acc = round(test_acc*100, 2)
        test_acc_list.append(round_acc)
        if (epoch + 1) % 5 == 0:
            print('\nEpoch[%d/%d] loss: %.4f, test acc: %.2f\n' % (epoch + 1, args.epoch, test_loss * 1.0 / count, round_acc)) 

    import csv
    with open(os.path.join(f'testacc_results.csv'), 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([args.poison_train, args.interval, args.dataset, args.target_model, round_acc])   


            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PointAPA: Towards Availability Poisoning Attacks in 3D Point Clouds')
    parser.add_argument('--batch_size', type=int, default=16, metavar='N', help='input batch size for training (default: 1)')
    parser.add_argument('--input_point_nums', type=int, default=1024, help='Point nums of each point cloud')
    parser.add_argument('--seed', type=int, default=2023, metavar='S', help='random seed (default: 2023)')
    parser.add_argument('--dataset', type=str, default='ModelNet10',  choices=['ModelNet10', 'ModelNet40', 'ShapeNetPart'])
    parser.add_argument('--NUM_CLASSES', type=int, default=10, help='The number of classes')
    parser.add_argument('--num_workers', type=int, default=4,help='Worker nums of data loading.')
    parser.add_argument('--target_model', type=str, default='pointnet_cls',choices=['pointnet_cls', 'pointnet2_cls_msg', 'dgcnn',  'pointcnn', 'pct'])
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',help='Num of nearest neighbors to use')
    parser.add_argument('--dropout', type=float, default=0.5,help='dropout rate')
    parser.add_argument('--epoch', default=80, type=int, help='')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--defense', action='store_true', help='using data augmentations to serve as defenses')
    parser.add_argument('--aug_type', type=str, default='rot') 
    parser.add_argument('--interval', type=int, default=42, help='The interval of rotation')
    parser.add_argument('--poison_train', action='store_true')

    args = parser.parse_args()
    set_seed(args.seed)
    args.device = torch.device("cuda")
    main()