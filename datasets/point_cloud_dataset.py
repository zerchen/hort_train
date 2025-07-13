#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :sdf_dataset.py
#@Date        :2022/04/05 16:58:35
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import os
import time
import cv2
import torch
import lmdb
import json
import copy
import random
import numpy as np
from glob import glob
from torch.utils.data.dataset import Dataset
from utils.camera import PerspectiveCamera
from base_dataset import BaseDataset
from kornia.geometry.conversions import rotation_matrix_to_axis_angle, axis_angle_to_rotation_matrix
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from utils.transform import dehomoify, homoify


class PointCloudDataset(BaseDataset):
    def __init__(self, db, cfg, mode='train'):
        super(PointCloudDataset, self).__init__(db, cfg, mode)
        self.num_sample_points = cfg.num_sample_points
        self.recon_scale = cfg.recon_scale
        self.test_with_gt_hand = cfg.test_with_gt_hand
        self.use_seg = cfg.use_seg
        self.use_3d_aug = cfg.use_3d_aug

    def __getitem__(self, index):
        sample_data = copy.deepcopy(self.db[index])

        sample_key = sample_data['id']
        img_path = sample_data['img_path']
        seg_path = sample_data['seg_path']
        bbox = sample_data['bbox']

        if self.mode == "train":
            right_hand_palm = torch.from_numpy(sample_data['hand_palm'])
        else:
            if self.dataset_name in ["dexycb", "obman", "mow", "hoi4d", "oakink", "hocap", "hograspnet"]:
                right_hand_palm = torch.from_numpy(sample_data['hand_palm'])
            elif self.dataset_name in ["ho3dv2", "ho3dv3", "core50", "zerchen"]:
                right_hand_palm = torch.zeros(3)
        if self.dataset_name in ["core50", "zerchen"]:
            right_hand_joints_3d = torch.zeros((21, 3))
        else:
            right_hand_joints_3d = torch.from_numpy(sample_data['hand_joints_3d'])

        if self.dataset_name in ["dexycb", "ho3dv2", "ho3dv3", "hoi4d", "mow", "core50", "zerchen", "oakink", "hocap", "hograspnet"]:
            if self.mode == "train":
                right_hand_poses = torch.from_numpy(sample_data['hand_poses'])
                right_hand_poses = axis_angle_to_rotation_matrix(right_hand_poses.reshape((-1, 3)))
                if self.dataset_name in ["mow"]:
                    right_hand_verts_3d = torch.from_numpy(sample_data['hand_verts_3d'])
                elif self.dataset_name in ["dexycb", "ho3dv2", "ho3dv3", "hoi4d", "oakink", "hocap", "hograspnet"]:
                    right_hand_verts_3d = torch.from_numpy(np.load(sample_data['verts_hand_path']).astype(np.float32))

                obj_transform = torch.from_numpy(sample_data['obj_transform'])
                object_mesh = load_objs_as_meshes([sample_data['mesh_obj_path']], load_textures=False, create_texture_atlas=False)
                point_clouds = sample_points_from_meshes(object_mesh, 2048)[0]
                point_clouds_up = sample_points_from_meshes(object_mesh, 16384)[0]
                if self.dataset_name in ["mow"]:
                    point_clouds = point_clouds - obj_transform[:3, 3]
                    point_clouds_up = point_clouds_up - obj_transform[:3, 3]
            else:
                if self.dataset_name in ["dexycb", "mow", "hoi4d"]:
                    try:
                        right_hand_poses = torch.from_numpy(sample_data['hand_poses'])
                        right_hand_poses = axis_angle_to_rotation_matrix(right_hand_poses.reshape((-1, 3)))
                    except:
                        right_hand_poses = torch.zeros((16, 3, 3))
                    if self.dataset_name in ["mow"]:
                        right_hand_verts_3d = torch.from_numpy(sample_data['hand_verts_3d'])
                    elif self.dataset_name in ["dexycb", "hoi4d", "oakink", "hocap", "hograspnet"]:
                        right_hand_verts_3d = torch.from_numpy(np.load(sample_data['verts_hand_path']).astype(np.float32))
                elif self.dataset_name in ["ho3dv2", "ho3dv3", "core50", "zerchen"]:
                    right_hand_poses = torch.zeros((16, 3, 3))
                    right_hand_verts_3d = torch.zeros((778, 3))

                try:
                    wilor_pred_cam_t = torch.from_numpy(sample_data['wilor_pred_cam_t'])
                    wilor_hand_joints_3d = torch.from_numpy(sample_data['wilor_mano_hand_joints_3d']) + wilor_pred_cam_t
                    wilor_hand_verts_3d = torch.from_numpy(sample_data['wilor_mano_hand_verts_3d']) + wilor_pred_cam_t
                    wilor_hand_palm = torch.from_numpy(sample_data['wilor_mano_hand_palm']) + wilor_pred_cam_t
                    # wilor_hand_shapes = torch.from_numpy(sample_data['wilor_hand_shapes'])
                    # wilor_hand_poses = torch.from_numpy(sample_data['wilor_hand_poses'])
                except:
                    pass
        else:
            right_hand_verts_3d = torch.from_numpy(np.load(sample_data['verts_hand_path']).astype(np.float32))
            right_hand_poses = torch.from_numpy(sample_data['hand_poses'])
            right_hand_poses = axis_angle_to_rotation_matrix(right_hand_poses.reshape((-1, 3)))
            obj_transform = torch.from_numpy(sample_data['obj_transform'])
            object_mesh = load_objs_as_meshes([sample_data['mesh_obj_path']], load_textures=False, create_texture_atlas=False)
            point_clouds = sample_points_from_meshes(object_mesh, 2048)[0] - obj_transform[:3, 3]
            point_clouds_up = sample_points_from_meshes(object_mesh, 16384)[0] - obj_transform[:3, 3]

        img = self.load_img(img_path)

        if sample_data['hand_side'] == 0:
            img = cv2.flip(img, 1)
            img_width = img.shape[1]
            bbox[0] = img_width - (bbox[0] + bbox[2])
            right_hand_palm[0] *= -1.
            right_hand_joints_3d[:, 0] *= -1.
            right_hand_verts_3d[:, 0] *= -1.
            if self.mode == "train":
                ref_mat = torch.eye(4)
                ref_mat[0, 0] = -1.
                obj_transform = ref_mat @ obj_transform @ ref_mat
                point_clouds[:, 0] *= -1.
                point_clouds_up[:, 0] *= -1.

        if self.mode == 'train':
            if sample_data['hand_side'] == 1:
                camera = PerspectiveCamera(sample_data['fx'], sample_data['fy'], sample_data['cx'], sample_data['cy'])
            else:
                camera = PerspectiveCamera(sample_data['fx'], sample_data['fy'], img_width - sample_data['cx'], sample_data['cy'])
            trans, scale, rot, do_flip, color_scale, scale_3d = self.get_aug_config(self.dataset_name)
            if not self.use_3d_aug:
                scale_3d = 1.
            rot_aug_mat = torch.from_numpy(np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]], dtype=np.float32))
        else:
            if self.dataset_name == "obman" or self.test_with_gt_hand:
                if sample_data['hand_side'] == 1:
                    camera = PerspectiveCamera(sample_data['fx'], sample_data['fy'], sample_data['cx'], sample_data['cy'])
                else:
                    camera = PerspectiveCamera(sample_data['fx'], sample_data['fy'], img_width - sample_data['cx'], sample_data['cy'])
            else:
                camera = PerspectiveCamera(5000 / 256 * self.input_image_size[0], 5000 / 256 * self.input_image_size[1], self.input_image_size[0] / 2, self.input_image_size[1] / 2)
            trans, scale, rot, do_flip, color_scale, scale_3d = [0, 0], 1, 0, False, [1.0, 1.0, 1.0], 1.0

        bbox[0] = bbox[0] + bbox[2] * trans[0]
        bbox[1] = bbox[1] + bbox[3] * trans[1]
        img, _ = self.generate_patch_image(img, bbox, self.input_image_size, do_flip, scale, rot)

        for i in range(3):
            img[:, :, i] = np.clip(img[:, :, i] * color_scale[i], 0, 255)

        if self.use_seg and self.dataset_name not in ["oakink", "hocap", "ho3dv3", "ho3dv2", "hograspnet"]:
            object_id = sample_data['ycb_id'] if self.dataset_name in ["dexycb"] else None
            seg = self.load_seg(seg_path, object_id)
            seg, _ = self.generate_patch_image(seg, bbox, self.input_image_size, do_flip, scale, rot)
            seg = seg[:, :, [0]]
            img = seg * img
            img = img.astype(np.uint8)

        if self.mode == 'train':
            camera.update_virtual_camera_after_crop(bbox)
            camera.update_intrinsics_after_resize((bbox[-1], bbox[-2]), self.input_image_size)
            camera.update_intrinsics_after_scale(scale)
            rot_cam_extr = torch.from_numpy(camera.extrinsics[:3, :3].T)
            rot_aug_mat = rot_aug_mat @ rot_cam_extr

            if self.use_inria_aug and random.random() < 0.5 and self.dataset_name in ["obman", "dexycb", "hoi4d"] and not self.use_seg:
                bg_list_path = glob(f"{self.inria_aug_source}/*.jpg")
                bg_img_path = random.choice(bg_list_path)
                bg = self.load_img(bg_img_path)
                bg = self.random_crop(bg, 224, 224)

                object_id = sample_data['ycb_id'] if self.dataset_name in ["dexycb"] else None
                seg = self.load_seg(seg_path, object_id)
                if sample_data['hand_side'] == 0:
                    seg = np.flip(seg, 1)
                seg, _ = self.generate_patch_image(seg, bbox, self.input_image_size, do_flip, scale, rot)
                seg = seg[:, :, [0]]
                img = seg * img + (1 - seg) * bg
                img = img.astype(np.uint8)
        else:
            if self.dataset_name == "obman" or self.test_with_gt_hand:
                camera.update_virtual_camera_after_crop(bbox)
                camera.update_intrinsics_after_resize((bbox[-1], bbox[-2]), self.input_image_size)
                camera.update_intrinsics_after_scale(scale)
            rot_cam_extr = torch.from_numpy(camera.extrinsics[:3, :3].T)

        input_img = self.image_transform(img)
        cam_intr = torch.from_numpy(camera.intrinsics)
        cam_extr = torch.from_numpy(camera.extrinsics)

        if self.mode == 'train':
            right_hand_joints_3d[:, 0:3] = torch.mm(rot_aug_mat, right_hand_joints_3d[:, 0:3].transpose(1, 0)).transpose(1, 0) * scale_3d
            right_hand_verts_3d[:, 0:3] = torch.mm(rot_aug_mat, right_hand_verts_3d[:, 0:3].transpose(1, 0)).transpose(1, 0) * scale_3d
            right_hand_palm = rot_aug_mat @ right_hand_palm * scale_3d
            right_hand_poses[0] = rot_aug_mat @ right_hand_poses[0]
            obj_transform[:3, :3] = rot_aug_mat @ obj_transform[:3, :3]
            obj_transform[:3, 3] = rot_aug_mat @ obj_transform[:3, 3] * scale_3d
            if self.dataset_name in ["dexycb", "ho3dv2", "ho3dv3", "hoi4d", "oakink", "hocap", "hograspnet"]:
                point_clouds = (obj_transform[:3, :3] @ point_clouds.transpose(1, 0)).transpose(1, 0) / self.recon_scale * scale_3d
                point_clouds_up = (obj_transform[:3, :3] @ point_clouds_up.transpose(1, 0)).transpose(1, 0) / self.recon_scale * scale_3d
            elif self.dataset_name in ["obman", "mow"]:
                point_clouds = (rot_aug_mat @ point_clouds.transpose(1, 0)).transpose(1, 0) / self.recon_scale * scale_3d
                point_clouds_up = (rot_aug_mat @ point_clouds_up.transpose(1, 0)).transpose(1, 0) / self.recon_scale * scale_3d

            right_hand_palm_2d = cam_intr[:3, :3] @ right_hand_palm
            right_hand_palm_2d = right_hand_palm_2d[:2] / right_hand_palm_2d[-1]
            right_hand_palm_2d = right_hand_palm_2d / self.input_image_size[0]

            obj_2d = cam_intr[:3, :3] @ obj_transform[:3, 3]
            obj_2d = obj_2d[:2] / obj_2d[-1]
            obj_2d = obj_2d / self.input_image_size[0]

            # t0 = (camera.intrinsics @ homoify(right_hand_verts_3d).numpy().transpose(1, 0)).transpose(1, 0)
            # t1 = (t0[:, :2] / t0[:, [2]]).astype(np.int32)
            # for i in range(t1.shape[0]):
                # if t1[i,0] > 0 and t1[i,0] < 224 and t1[i,1] > 0 and t1[i,1] < 224:
                    # cv2.circle(img, t1[i], 1, (0, 0, 255), 4)
            # cv2.imwrite('test.jpg', img[:, :, ::-1])

            input_iter = dict(img=input_img)
            label_iter = dict(point_clouds=point_clouds, point_clouds_up=point_clouds_up)
            meta_iter = dict(cam_intr=cam_intr, cam_extr=cam_extr, id=sample_key, right_hand_joints_3d=right_hand_joints_3d, right_hand_verts_3d=right_hand_verts_3d, right_hand_palm=right_hand_palm, obj_transform=obj_transform)

            return input_iter, label_iter, meta_iter
        else:
            input_iter = dict(img=input_img)

            if self.dataset_name in ["obman"] or self.test_with_gt_hand:
                right_hand_joints_3d[:, 0:3] = torch.mm(rot_cam_extr, right_hand_joints_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)
                right_hand_verts_3d[:, 0:3] = torch.mm(rot_cam_extr, right_hand_verts_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)
                right_hand_palm = rot_cam_extr @ right_hand_palm
                right_hand_poses[0] = rot_cam_extr @ right_hand_poses[0]
                meta_iter = dict(cam_intr=cam_intr, cam_extr=cam_extr, id=sample_key, right_hand_joints_3d=right_hand_joints_3d, right_hand_verts_3d=right_hand_verts_3d, right_hand_palm=right_hand_palm, img=img)
            else:
                meta_iter = dict(cam_intr=cam_intr, cam_extr=cam_extr, id=sample_key, right_hand_joints_3d=wilor_hand_joints_3d, right_hand_verts_3d=wilor_hand_verts_3d, right_hand_palm=wilor_hand_palm, img=img)

            return input_iter, meta_iter


if __name__ == "__main__":
    from obman.obman import obman
    obman_db = obman('train_30k')
