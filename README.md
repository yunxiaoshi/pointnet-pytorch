## PointNet-PyTorch

This is a PyTorch implementation of [PointNet (CVPR 2017)](https://arxiv.org/abs/1612.00593 "PointNet"), with comprehensive experiments.

## Installation

Refer to requirements.txt for common dependencies which are fairly easy to install via conda or pip. You may also need to install [PyMesh](https://github.com/PyMesh/PyMesh "PyMesh"). See [here](https://github.com/PyMesh/PyMesh#Build) for instructions to install.

The code supports Python3 and PyTorch 0.4.1+.

## Usage

This code implements object classification on ModelNet, shape part segmentation on ShapeNet and indoor scene semantic segmentation on the Stanford 3D dataset.

### ModelNet Classification

Download the ModelNet10 dataset from [here](http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip) or the ModelNet40 dataset from [here](https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_manually_aligned.tar). Unzip and run 
```
python train_cls.py -dset modelnet40 -r modelnet_root_dir -np number_of_points_to_sample
```

### ShapeNet Part Segmentation

Download the ShapeNet dataset from [here](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip). Unzip and run
```
python train_seg.py -dset shapenet16 -r shapenet_root_dir -np number_of_points_to_sample
```

### Indoor Scene Semantic Segmentation

Download the S3DIS dataset from [here](http://buildingparser.stanford.edu/dataset.html#Download) (you need to submit a request). Unzip and do
```
cd Stanford3dDataset_v1.2
mkdir train test
mv Area_1 Area_2 Area_3 Area_4 Area_6 train
mv Area_5 test
```
to create train/test split. Then set ```gen_labels=True``` in the class ```S3dDataset``` in datasets.py and do
```
python datasets.py
``` 
to generate labels for the train and test set respectively. __After that always set ```gen_labels=False```__. With labels generated do
```
python train_seg.py -dset s3dis -r s3dis_root_dir -np number_of_points_to_sample
```
to start training.

## Visualization

First do ```sh build.sh```, then use ```show_seg.py``` to visualize segmented object parts. Below are some example results.

![](https://i.ibb.co/Vppscz6/part.png "part results")

For S3DIS, you have to combine scene components along with their labels into one text file and then pass it to ```show_seg_s3dis.py```. Below are some example results (removed some clutter classes for better visualization).

![](https://i.ibb.co/tYHszTC/s3dis.png "s3dis results")

## Results

Certain design choices in the original paper are not implemented here for simplicity. There is some performance gap on ModelNet classification, for ShapeNet and S3DIS seems to be on par with the original paper.

| | accuracy| 
| :------: | :------: |
| ModelNet10 | 87.2% |
| ModelNet40 | 85.4% |

| | accuracy | class avg IoU
| :------: | :------: | :------: |
| ShapeNet | - | 82.9%|

| | accuracy | class avg IoU|
| :------: | :------: | :------: |
| S3DIS | 72.1% | 50.6% |

## Acknowledgements

[pointnet.pytorch](https://github.com/fxia22/pointnet.pytorch) (many thanks)

[original tensorflow implementation](https://github.com/charlesq34/pointnet)


