# -*- coding: utf-8 -*-

import os

def modelnet_cat2num(modelnet_root):
    for i, item in enumerate(os.listdir(modelnet_root)):
        with open(os.path.join(modelnet_root, 'modelnet_cat2num.txt'), 'a') as f:
            f.write(item + ' ' + str(i) + '\n')


shapenet_labels = {'Airplane': 4, 
        'Bag': 2, 
        'Cap': 2, 
        'Car': 4, 
        'Chair': 4, 
        'Earphone': 3, 
        'Guitar': 3, 
        'Knife': 2, 
        'Lamp': 4, 
        'Laptop': 2, 
        'Motorbike': 6, 
        'Mug': 2, 
        'Pistol': 3, 
        'Rocket': 3, 
        'Skateboard': 3, 
        'Table': 3
        }

def s3d_cat2num(s3d_root):
    for item in s3d_cat2num:
        with open(os.path.join(s3d_root, 's3d_cat2num.txt'), 'a') as f:
            f.write(item + ' ' + str(s3d_cat2num[item]) + '\n')
            

if __name__ == '__main__':

    '''
    modelnet_root = './ModelNet10'
    modelnet_cat2num(modelnet_root)

    s3d_root = './Stanford3dDataset_v1.2'
    s3d_cat2num(s3d_root)
    '''
