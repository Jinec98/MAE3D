#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import glob
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

def download_modelnet40():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % ('modelnet40_ply_hdf5_2048', DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data_cls(partition):
    download_modelnet40()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', '*%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r+')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def load_data_shapenet55():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    filename = os.path.join(DATA_DIR, 'shapenet57448xyzonly.npz')
    data_dict = dict(np.load(filename))
    all_data = data_dict['data']
    all_index = np.arange(len(all_data))

    return all_data, all_index


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(rotation_matrix)  # random rotation (x,z)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, limited_ratio=1.0, partition='train'):
        self.data, self.label = load_data_cls(partition)
        self.data_size = self.data.shape[0]
        self.limited_data_size = int(self.data_size * limited_ratio)
        self.data = self.data[:self.limited_data_size]
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ShapeNet55(Dataset):
    def __init__(self, num_points):
        self.data, self.index = load_data_shapenet55()
        self.num_points = num_points

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        index = self.index[item]

        pointcloud = translate_pointcloud(pointcloud)
        np.random.shuffle(pointcloud)
        return pointcloud, index

    def __len__(self):
        return self.data.shape[0]


class ScanObjectNN(Dataset):
    def __init__(self, num_points, root, partition='train'):
        super().__init__()
        self.num_points = num_points
        self.partition = partition
        self.root = root

        if self.partition == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.partition == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        if self.partition == 'train':
            np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()

        if self.partition == 'train':
            current_points = translate_pointcloud(current_points)
            # np.random.shuffle(current_points)

        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]

        return current_points, label

    def __len__(self):
        return self.points.shape[0]


class ScanObjectNN_hardest(Dataset):
    def __init__(self, num_points, root, partition='train'):
        super().__init__()
        self.num_points = num_points
        self.partition = partition
        self.root = root

        if self.partition == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.partition == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        if self.partition == 'train':
            np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()

        if self.partition == 'train':
            current_points = translate_pointcloud(current_points)
            # np.random.shuffle(current_points)

        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]

        return current_points, label

    def __len__(self):
        return self.points.shape[0]


if __name__ == '__main__':
    train = ShapeNet55(2048)
    data = train[0]
    print(data.shape)

