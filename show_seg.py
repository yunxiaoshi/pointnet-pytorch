# -*- coding: utf-8 -*-

import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from datasets import PartDataset
from pointnet import PointNetSeg
from utils import shapenet_labels
from show3d_balls import *

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('--className', type=str, default='Chair', help='number of classes')
parser.add_argument('--radius', type=int, default=2, help='radius of ball for visualization')
parser.add_argument('--cmap', type=str, default='hsv', help='valid matplotlib cmap')
parser.add_argument('--npoints', type=int, default=2500, help='points to sample')

opt = parser.parse_args()

idx = opt.idx

d = PartDataset(root='shapenetcore_partanno_segmentation_benchmark_v0', class_choice=[opt.className], train=False, npoints=2048)

print('model %d/%d' % (idx, len(d)))

num_class = d.num_classes
print('number of classes', num_class)

point, seg = d[idx]

point_np = point.numpy()

cmap = plt.cm.get_cmap(opt.cmap, 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
gt = cmap[seg.numpy()-1, :]

classifier = PointNetDenseCls(k=shapenet_labels[opt.className])
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

point = point.transpose(1, 0).contiguous()
point = point.view(1, point.size()[0], point.size()[1])


pred, _ = classifier(point)

pred_choice = pred.data.max(2)[1]

pred_color = cmap[pred_choice.numpy()[0], :]

showpoints(point_np, gt, gt, ballradius=opt.radius)


