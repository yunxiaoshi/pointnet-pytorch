# -*- coding: utf-8 -*-

import argparse
import os
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lera

from datasets import ClsDataset
from pointnet import PointNetCls


def train(config):
    print('Random seed: %d' % int(config.seed))
    torch.manual_seed(config.seed)
    
    torch.backends.cudnn.benchmark = True

    dset = config.dataset
    if dset == 'modelnet10' or dset == 'modelnet40':
        dataset = ClsDataset(root=config.root, npoints=config.npoints, train=True)
        test_dataset = ClsDataset(root=config.root, npoints=config.npoints, train=False)
    else:
        raise NotImplementedError('Dataset not supported.')
    
    print('Selected %s' % dset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batchsize, shuffle=True, 
                num_workers=config.workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batchsize, shuffle=True, 
        num_workers=config.workers)

    num_classes = dataset.num_classes
    print('number of classes: %d' % num_classes)
    print('train set size: %d | test set size: %d' % (len(dataset), len(test_dataset)))
    try:
        os.makedirs(config.outf)
    except:
        pass

    blue = lambda x: '\033[94m' + x + '\033[0m'
    yellow = lambda x: '\033[93m' + x + '\033[0m'
    red = lambda x: '\033[91m' + x + '\033[0m'

    classifier = PointNetCls(k=num_classes)

    if config.model != '':
        classifier.load_state_dict(torch.load(config.model))

    optimizer = optim.SGD(classifier.parameters(), lr=config.lr, momentum=config.momentum)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier.to(device)
    if config.mgpu:
        classifier = torch.nn.DataParallel(classifier, device_ids=config.gpuids)

    num_batch = len(dataset) / config.batchsize

    lera.log_hyperparams({
        'title': dset, 
        'batchsize': config.batchsize, 
        'epochs': config.nepochs, 
        'npoints': config.npoints, 
        'optimizer': 'SGD', 
        'lr': config.lr, 
        })

    for epoch in range(config.nepochs):
        train_acc_epoch, test_acc_epoch = [], []
        for i, data in enumerate(dataloader):
            points, labels = data
            points = points.transpose(2, 1)
            labels = labels[:, 0]
            points, labels = points.to(device), labels.to(device)
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, _ = classifier(points)
            pred = pred.view(-1, num_classes)
            # print(pred.size(), labels.size())
            loss = F.nll_loss(pred, labels)
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(labels.data).cpu().sum()
            train_acc = correct.item() / float(config.batchsize)
            print('epoch %d: %d/%d | train loss: %f | train acc: %f' % (epoch+1, i+1, num_batch+1, loss.item(), train_acc))
            train_acc_epoch.append(train_acc)
            lera.log({
                'train loss': loss.item(), 
                'train acc': train_acc
                })

            if (i+1) % 10 == 0:
                j, data = next(enumerate(test_dataloader, 0))
                points, labels = data
                points = points.transpose(2, 1)
                labels = labels[:, 0]
                points, labels = points.to(device), labels.to(device)
                classifier = classifier.eval()
                with torch.no_grad():
                    pred, _ = classifier(points)
                pred = pred.view(-1, num_classes)
                loss = F.nll_loss(pred, labels)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(labels.data).cpu().sum()
                test_acc = correct.item() / float(config.batchsize)
                print(blue('epoch %d: %d/%d | test loss: %f | test acc: %f') % (epoch+1, i+1, num_batch+1, loss.item(), test_acc))
                test_acc_epoch.append(test_acc)
                lera.log({
                    'test loss': loss.item(), 
                    'test acc': test_acc
                    })
        print(yellow('epoch %d | mean train acc: %f') % (epoch+1, np.mean(train_acc_epoch)))
        print(red('epoch %d | mean test acc: %f') % (epoch+1, np.mean(test_acc_epoch)))
        lera.log({
            'train acc epoch': np.mean(train_acc_epoch), 
            'test acc epoch': np.mean(test_acc_epoch)})
        torch.save(classifier.state_dict(), '%s/%s_model_%d.pth' % (config.outf, config.dataset, epoch))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, help='random seed')
    parser.add_argument('-dset', '--dataset', type=str, required=True, help='dataset to train on, one of modelnet10 and modelnet40)
    parser.add_argument('-r', '--root', type=str, required=True, help='path to dataset')
    parser.add_argument('-np', '--npoints', type=int, help='number of points to sample')
    parser.add_argument('-bs', '--batchsize', type=int, default=32, help='batch size')
    parser.add_argument('-ws', '--workers', type=int, default=4, help='number of workers')
    parser.add_argument('-out', '--outf', type=str, default='./checkpoints', help='path to save model checkpoints')
    parser.add_argument('--model', type=str, default='', help='checkpoint dir')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--mgpu', type=bool, default=False, help='whether to utilize multiple gpus')
    parser.add_argument('--gpuids', nargs='+', type=int, help='which gpus to use')
    parser.add_argument('--nepochs', type=int, default=100, help='epochs to train')
    config = parser.parse_args()

    train(config)


