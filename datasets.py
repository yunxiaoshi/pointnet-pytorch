# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
import torch.utils.data as data
import pymesh
from tqdm import tqdm
from utils import shapenet_labels

def scale_linear_bycolumn(rawdata, high=1.0, low=0.0):
    mins = np.min(rawdata, axis=0)
    maxs = np.max(rawdata, axis=0)
    rng = maxs - mins
    return high - (high-low)*(maxs-rawdata)/(rng+np.finfo(np.float32).eps)


class ClsDataset(data.Dataset):
        '''Object classification on ModelNet'''
        def __init__(self, root, npoints=1024, train=True):
                self.root = root
                self.npoints = npoints
                self.catfile = os.path.join(self.root, 'modelnet_cat2num.txt')
                self.cat = {}

                with open(self.catfile, 'r') as f:
                        for line in f.readlines():
                                lns = line.strip().split()
                                self.cat[lns[0]] = lns[1]
                self.num_classes = len(self.cat)
                self.datapath = []
                FLAG = 'train' if train else 'test'
                for item in os.listdir(self.root):
                        if os.path.isdir(os.path.join(self.root, item)):
                                for f in os.listdir(os.path.join(self.root, item, FLAG)):
                                    if f.endswith('.off'):
                                        self.datapath.append((os.path.join(self.root, item, FLAG, f), int(self.cat[item])))


        def __getitem__(self, idx):
                fn = self.datapath[idx]
                points = pymesh.load_mesh(fn[0]).vertices
                label = fn[1]
                replace = True if points.shape[0]<self.npoints else False
                choice = np.random.choice(points.shape[0], self.npoints, replace=replace)
                points = points[choice, :]
                points = scale_linear_bycolumn(points)
                points = torch.from_numpy(points.astype(np.float32))
                label = torch.from_numpy(np.array([label]).astype(np.int64))
                return points, label


        def __len__(self):
                return len(self.datapath)


class PartDataset(data.Dataset):
        def __init__(self, root, npoints=2048, class_choice=None, train=True):
                '''Part segmentation on ShapeNet'''
                self.root = root
                self.npoints = npoints
                self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
                self.cat = {}

                with open(self.catfile, 'r') as f:
                        for line in f.readlines():
                                lns = line.strip().split()
                                self.cat[lns[0]] = lns[1]
                
                if not class_choice is None:
                    self.cat = {k:v for k, v in self.cat.items() if k in class_choice}
                    self.num_classes = shapenet_labels[class_choice[0]]
                else:
                    self.num_classes = 50

                self.meta = {}
                for item in self.cat:
                        self.meta[item] = []
                        point_dir = os.path.join(self.root, self.cat[item], 'points')
                        seg_dir = os.path.join(self.root, self.cat[item], 'points_label')
                fns = sorted(os.listdir(point_dir))
                if train:
                        fns = fns[:int(0.9*len(fns))]
                else:
                        fns = fns[int(0.9*len(fns)):]

                for fn in fns:
                        token = (os.path.splitext(os.path.basename(fn))[0])
                        self.meta[item].append((os.path.join(point_dir, token + '.pts'), os.path.join(seg_dir, token + '.seg')))

                self.datapath = []
                for item in self.cat:
                        for fn in self.meta[item]:
                                self.datapath.append((item, fn[0], fn[1]))

                self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))


        def __getitem__(self, idx):
                fn = self.datapath[idx]
                points = np.loadtxt(fn[1]).astype(np.float32)
                seg = np.loadtxt(fn[2]).astype(np.int64)
                replace = True if points.shape[0]<self.npoints else False
                choice = np.random.choice(len(seg), self.npoints, replace=replace)
                # resample
                points = points[choice, :]
                seg = seg[choice]
                points = torch.from_numpy(points)
                seg = torch.from_numpy(seg)
                return points, seg


        def __len__(self):
                return len(self.datapath)


class S3dDataset(data.Dataset):
        '''Semantic segmentation on S3DIS'''
        def __init__(self, root, npoints=4096, train=True, gen_labels=False):
                self.root = root
                self.npoints = npoints
                self.catfile = os.path.join(self.root, 's3d_cat2num.txt')
                self.cat = {}
                with open(self.catfile, 'r') as f:
                    for line in f.readlines():
                        lns = line.strip().split()
                        self.cat[lns[0]] = lns[1]
                self.num_classes = len(self.cat)
                self.datapath, self.labelspath = [], []
                FLAG = 'train' if train else 'test'
                path = os.path.join(self.root, FLAG)
                for area in os.listdir(path):
                    area_path = os.path.join(path, area)
                    for scene in os.listdir(area_path):
                        if os.path.isdir(os.path.join(area_path, scene)):
                            scene_path = os.path.join(area_path, scene)
                            for scene_component in os.listdir(os.path.join(scene_path, 'Annotations')):
                                if not scene_component.endswith('_labels.txt'):
                                    self.datapath.append(os.path.join(scene_path, 'Annotations', scene_component))
                

                if gen_labels: # do this only once
                    pbar = tqdm(total=len(self.datapath))
                    for path in self.datapath:
                        l = path.split('/')
                        labels_path = os.path.join(l[0], l[1], l[2], l[3], l[4])
                        component_name = l[-1].split('.')[0]
                        class_name = l[-1].split('_')[0]
                        with open(path, 'r') as f:
                            for line in f.readlines():
                                with open(os.path.join(labels_path, component_name + '_labels.txt'), 'a') as g:
                                    g.write(str(self.cat[class_name]) + '\n')
                        pbar.update()
                    

        def __getitem__(self, idx):
            fn = self.datapath[idx]
            points = np.loadtxt(fn)[:, :3].astype(np.float32)
            ln = os.path.splitext(fn)[0] + '_labels.txt'
            seg = np.loadtxt(ln).astype(np.int64)
            replace = True if points.shape[0]<self.npoints else False
            choice = np.random.choice(points.shape[0], self.npoints, replace=replace)
            points = points[choice, :]
            points = scale_linear_bycolumn(points)
            seg = seg[choice]
            points = torch.from_numpy(points)
            seg = torch.from_numpy(seg)
            return points, seg


        def __len__(self):
            return len(self.datapath)


if __name__ == '__main__':

        c = ClsDataset(root='modelnet40_manually_aligned')
        print(ps.type(), ps.size(), l.type(), l.size(), l)

        d = PartDataset(root='shapenetcore_partanno_segmentation_benchmark_v0')
        ps, seg = d[10]
        print(ps.type(), ps.size(), seg.type(), seg.size())

        s = S3dDataset(root='Stanford3dDataset_v1.2', train=False, gen_labels=False)
        ps, seg = s[100]
        print(ps.type(), ps.size(), seg.type(), seg.size())



