#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :base_dataset.py
#@Date        :2022/04/05 16:58:51
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import os
import time
import cv2
import torch
import lmdb
import copy
import numpy as np
from torchvision import transforms
from utils.random_utils import np_random
from torch.utils.data.dataset import Dataset
from utils.img_utils import parse_2Dmask_img


class BaseDataset(Dataset):
    def __init__(self, db, cfg, mode='train'):
        self.mode = mode
        self.dataset_name = db.name
        self.stage = db.stage_split
        self.db = db.data
        self.joints_name = db.joints_name
        self.inria_aug_source = db.inria_aug_source

        self.use_lmdb = cfg.use_lmdb
        self.hand_branch = cfg.hand_branch
        self.obj_branch = cfg.obj_branch
        self.original_image_size = db.image_size
        self.input_image_size = cfg.image_size
        self.use_inria_aug = cfg.use_inria_aug
        self.np_random, _ = np_random()
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        if self.use_lmdb and self.mode == 'train':
            self.img_source = db.img_source + '.lmdb'
            self.img_env = lmdb.open(self.img_source, readonly=True, lock=False, readahead=False, meminit=False)

        if self.use_lmdb and self.use_inria_aug and self.mode == 'train':
            self.seg_env = lmdb.open(db.seg_source + '.lmdb', readonly=True, lock=False, readahead=False, meminit=False)
            self.inria_env = lmdb.open(db.inria_aug_source + '.lmdb', readonly=True, lock=False, readahead=False, meminit=False)
    
    def __len__(self):
        return len(self.db)
    
    def __getitem__(self, index):
        pass
    
    def load_img_lmdb(self, env, key, size, order='RGB'):
        """
        @description: load images from lmdb datasets.
        ---------
        @param: image lmdb env, data_key, image_size, channel order
        -------
        @Returns: image tensor in RGB order (default)
        -------
        """

        with env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        img_flat = np.frombuffer(buf, dtype=np.uint8)
        C, H, W = size
        img = img_flat.reshape(H, W, C)

        if order=='RGB':
            img = img[:,:,::-1].copy()

        return img
    
    def load_img(self, path, order='RGB'):
        """
        @description: load images directly from the disk
        ---------
        @param: image path, channel order
        -------
        @Returns: image tensor in RGB order (default)
        -------
        """

        img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not isinstance(img, np.ndarray):
            raise IOError("Fail to read %s" % path)

        if order=='RGB':
            img = img[:,:,::-1].copy()

        img = img.astype(np.uint8)
        return img

    def load_seg_lmdb(self, env, key, size, ycb_id):
        with env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        img_flat = np.frombuffer(buf, dtype=np.uint8)
        C, H, W = size
        img = img_flat.reshape(H, W, C)

        seg_maps = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
        if self.dataset_name == 'obman':
            seg_maps[:, :, 0][np.where(img[:, :, 0] == 100)] = 1
            seg_maps[:, :, 1][np.where(img[:, :, 0] == 22)] = 1
            seg_maps[:, :, 1][np.where(img[:, :, 0] == 24)] = 1
        elif self.dataset_name == 'dexycb':
            seg_maps[:, :, 0][np.where(img[:, :, 0] == ycb_id)] = 1
            seg_maps[:, :, 1][np.where(img[:, :, 0] == 255)] = 1

        return seg_maps

    def load_seg(self, path, ycb_id):
        # the first channel is hand_obj seg, the second channel is obj seg
        if self.dataset_name == 'obman':
            img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            seg_maps = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
            seg_maps[:, :, 0][np.where((img[:, :, 0] == 20) | (img[:, :, 0] == 22) | (img[:, :, 0] == 24) | (img[:, :, 0] == 100))] = 1.0
            seg_maps[:, :, 1][np.where(img[:, :, 0] == 100)] = 1.0
        elif self.dataset_name == 'dexycb':
            img = np.load(path)['seg']
            seg_maps = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
            seg_maps[:, :, 0][np.where((img == ycb_id) | (img == 255))] = 1.0
            seg_maps[:, :, 1][np.where((img == ycb_id))] = 1.0
        elif self.dataset_name == 'hoi4d':
            img = cv2.imread(path)
            mask = parse_2Dmask_img(img)
            seg_maps = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.float32)
            seg_maps[:, :, 0][np.where((mask == 2) | (mask == 4))] = 1.0
            seg_maps[:, :, 1][np.where((mask == 4))] = 1.0

        return seg_maps

    def random_crop(self, image, crop_height, crop_width):
        max_x = image.shape[1] - crop_width
        max_y = image.shape[0] - crop_height
        x = np.random.randint(0, max_x)
        y = np.random.randint(0, max_y)
        crop = image[y: y + crop_height, x: x + crop_width]
        return crop

    def get_aug_config(self, dataset):
        """
        @description: Modfied from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                      Set augmentation configs for different datasets.
        ---------
        @param: the name of the dataset.
        -------
        @Returns: parameters for different augmentations, including rotation, flip and color.
        -------
        """

        if dataset == 'obman':
            trans_factor = 0.0
            scale_factor = 0.25
            rot_factor = 90.
            enable_flip = False
            color_factor = 0.2
            scale_3d_factor = [1.0, 1.25]
        elif dataset == 'dexycb':
            trans_factor = 0.10
            scale_factor = 0.25
            rot_factor = 90.
            enable_flip = False
            color_factor = 0.2
            scale_3d_factor = [0.8, 1.2]
        elif dataset in ['ho3dv2', 'ho3dv3']:
            trans_factor = 0.10
            scale_factor = 0.25
            rot_factor = 90.
            enable_flip = False
            color_factor = 0.2
            scale_3d_factor = [0.8, 1.2]
        elif dataset in ['hoi4d']:
            trans_factor = 0.10
            scale_factor = 0.25
            rot_factor = 90.
            enable_flip = False
            color_factor = 0.2
            scale_3d_factor = [0.8, 1.2]
        elif dataset == 'h2o3d':
            trans_factor = 0.0
            scale_factor = 0.0
            rot_factor = 15.
            enable_flip = False
            color_factor = 0.2
            scale_3d_factor = [0.8, 1.2]
        elif dataset == 'mow':
            trans_factor = 0.10
            scale_factor = 0.25
            rot_factor = 90.
            enable_flip = False
            color_factor = 0.2
            scale_3d_factor = [0.8, 1.0]
        elif dataset in ['oakink', 'hocap', 'hograspnet']:
            trans_factor = 0.10
            scale_factor = 0.25
            rot_factor = 90.
            enable_flip = False
            color_factor = 0.2
            scale_3d_factor = [0.8, 1.2]

        trans = [self.np_random.uniform(-trans_factor, trans_factor), self.np_random.uniform(-trans_factor, trans_factor)]
        scale = np.clip(self.np_random.randn(), -1.0, 1.0) * scale_factor + 1.0
        rot = np.clip(self.np_random.randn(), -2.0, 2.0) * rot_factor if self.np_random.random() <= 0.6 else 0

        if enable_flip:
            do_flip = self.np_random.random() <= 0.5
        else:
            do_flip = False

        c_up = 1.0 + color_factor
        c_low = 1.0 - color_factor
        color_scale = [self.np_random.uniform(c_low, c_up), self.np_random.uniform(c_low, c_up), self.np_random.uniform(c_low, c_up)]

        scale_3d = self.np_random.uniform(scale_3d_factor[0], scale_3d_factor[1])

        return trans, scale, rot, do_flip, color_scale, scale_3d

    def generate_patch_image(self, cvimg, bbox, input_shape, do_flip, scale, rot):
        """
        @description: Modified from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                      generate the patch image from the bounding box and other parameters.
        ---------
        @param: input image, bbox(x1, y1, h, w), dest image shape, do_flip, scale factor, rotation degrees.
        -------
        @Returns: processed image, affine_transform matrix to get the processed image.
        -------
        """

        img = cvimg.copy()
        img_height, img_width, _ = img.shape

        bb_c_x = float(bbox[0] + 0.5 * bbox[2])
        bb_c_y = float(bbox[1] + 0.5 * bbox[3])
        bb_width = float(bbox[2])
        bb_height = float(bbox[3])

        if do_flip:
            img = img[:, ::-1, :]
            bb_c_x = img_width - bb_c_x - 1

        trans = self.gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale, rot, inv=False)
        img_patch = cv2.warpAffine(img, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)
        new_trans = np.zeros((3, 3), dtype=np.float32)
        new_trans[:2, :] = trans
        new_trans[2, 2] = 1

        return img_patch, new_trans

    def gen_trans_from_patch_cv(self, c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
        """
        @description: Modified from https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE/blob/master/data/dataset.py.
                      get affine transform matrix
        ---------
        @param: image center, original image size, desired image size, scale factor, rotation degree, whether to get inverse transformation.
        -------
        @Returns: affine transformation matrix
        -------
        """

        def rotate_2d(pt_2d, rot_rad):
            x = pt_2d[0]
            y = pt_2d[1]
            sn, cs = np.sin(rot_rad), np.cos(rot_rad)
            xx = x * cs - y * sn
            yy = x * sn + y * cs
            return np.array([xx, yy], dtype=np.float32)

        # augment size with scale
        src_w = src_width * scale
        src_h = src_height * scale
        src_center = np.array([c_x, c_y], dtype=np.float32)

        # augment rotation
        rot_rad = np.pi * rot / 180
        src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
        src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

        dst_w = dst_width
        dst_h = dst_height
        dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
        dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
        dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = src_center
        src[1, :] = src_center + src_downdir
        src[2, :] = src_center + src_rightdir

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = dst_center
        dst[1, :] = dst_center + dst_downdir
        dst[2, :] = dst_center + dst_rightdir

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans
