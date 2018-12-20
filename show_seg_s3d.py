# -*- coding: utf-8 -*-

import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from pointnet import PointNetSeg
from datasets import S3dDataset
from datasets import scale_linear_bycolumn

from show3d_balls import *


def parse_whole_scene(scene_path, scene_num, npoints=4096):
    scene = np.loadtxt(scene_path).astype(np.float32)
    seg = scene[:, -1].astype(np.int64)
    scene = scene[:, :3]
    replace = False if (scene_num*npoints<scene.shape[0]) else True
    choice = np.random.choice(scene.shape[0], scene_num*npoints, replace=replace)
    scene = scene[choice, :]
    seg = seg[choice]
    scene = scale_linear_bycolumn(scene)
    scene = torch.from_numpy(scene)
    seg = torch.from_numpy(seg)
    return scene, seg

parser = argparse.ArgumentParser()

parser.add_argument('--scene_path', type=str, default='', help='path to scene')
parser.add_argument('--scene_num', type=int, default='', help='number of scene components')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--className', type=str, default='Chair', help='number of classes')
parser.add_argument('--radius', type=int, default=2, help='radius of ball for visualization')
parser.add_argument('--cmap', type=str, default='hsv', help='valid matplotlib cmap')
parser.add_argument('--npoints', type=int, default=2500, help='points to sample')

opt = parser.parse_args()

num_classes = 14

point, seg = parse_whole_scene(opt.scene_path, opt.scene_num, int(opt.npoints))

point_np = point.numpy()

cmap = plt.cm.get_cmap(opt.cmap, 14)
cmap = np.array([cmap(i) for i in range(14)])[:, :3]
gt = cmap[seg.numpy(), :]

classifier = PointNetSeg(k=num_classes)

state_dict = torch.load(opt.model)

'''
*** load multi-gpu model ***
from collections import OrderedDict

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:]
    new_state_dict[name] = v
'''

classifier.load_state_dict(state_dict)
classifier.eval()

point = point.transpose(1, 0).contiguous()
point = point.view(1, point.size()[0], point.size()[1])


pred, _ = classifier(point)

pred_choice = pred.data.max(2)[1]

pred_color = cmap[pred_choice.numpy()[0], :]

showpoints(point_np, gt, gt, ballradius=opt.radius)

