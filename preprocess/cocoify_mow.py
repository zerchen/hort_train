#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :cocoify_obman.py
#@Date        :2022/04/20 16:18:30
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import numpy as np
import torch
from torch import nn
import cv2
import os
import os.path as osp
from tqdm import tqdm
from fire import Fire
import json
import pickle
import scipy.linalg
import trimesh
import sys
sys.path.insert(0, '../common')
from mano.manolayer import ManoLayer
from scipy.spatial.transform import Rotation as R
from utils.img_utils import generate_patch_image, process_bbox


mano_faces = np.load('../common/mano/assets/closed_fmano.npy')
mano_layer = ManoLayer(ncomps=45, center_idx=0, side="right", mano_root='../common/mano/assets/', use_pca=False, flat_hand_mean=False)


def get_obj_corners(mesh_path, obj_transform):
    mesh = trimesh.load(mesh_path, process=False)
    verts = mesh.vertices

    inv_obj_transform = np.linalg.inv(obj_transform)
    homo_verts = np.concatenate([verts, np.ones((verts.shape[0], 1))], 1)
    homo_rest_verts = np.dot(inv_obj_transform, homo_verts.transpose(1, 0)).transpose(1, 0)
    rest_verts = homo_rest_verts[:, :3] / homo_rest_verts[:, [3]]

    min_verts = rest_verts.min(0)
    max_verts = rest_verts.max(0)

    obj_rest_corners = np.zeros((9, 3), dtype=np.float32)
    obj_rest_corners[0] = np.array([0., 0., 0.])
    obj_rest_corners[1] = np.array([min_verts[0], min_verts[1], min_verts[2]])
    obj_rest_corners[2] = np.array([min_verts[0], max_verts[1], min_verts[2]])
    obj_rest_corners[3] = np.array([max_verts[0], min_verts[1], min_verts[2]])
    obj_rest_corners[4] = np.array([max_verts[0], max_verts[1], min_verts[2]])
    obj_rest_corners[5] = np.array([min_verts[0], min_verts[1], max_verts[2]])
    obj_rest_corners[6] = np.array([min_verts[0], max_verts[1], max_verts[2]])
    obj_rest_corners[7] = np.array([max_verts[0], min_verts[1], max_verts[2]])
    obj_rest_corners[8] = np.array([max_verts[0], max_verts[1], max_verts[2]])

    homo_obj_rest_corners = np.concatenate([obj_rest_corners, np.ones((obj_rest_corners.shape[0], 1))], 1)
    homo_obj_corners = np.dot(obj_transform, homo_obj_rest_corners.transpose(1, 0)).transpose(1, 0)
    obj_corners = homo_obj_corners[:, :3] / homo_obj_corners[:, [3]]

    return obj_corners, obj_rest_corners


def preprocess(data_root='../datasets/mow', mode='train'):
    data_path = osp.join(data_root, 'data')
    mesh_hand_path = osp.join(data_root, 'data', 'mesh_hand')
    mesh_obj_path = osp.join(data_root, 'data', 'mesh_obj')
    rest_mesh_obj_path = osp.join(data_root, 'data', 'rest_mesh_obj')
    os.makedirs(mesh_hand_path, exist_ok=True)
    os.makedirs(mesh_obj_path, exist_ok=True)
    os.makedirs(rest_mesh_obj_path, exist_ok=True)

    with open(os.path.join(data_path, 'poses.json'), 'r') as f:
        meta_data = json.load(f)
    
    with open(os.path.join(data_path, f'{mode}_s0.json'), 'r') as f:
        data_list = json.load(f)

    selected_ids = []
    with open(f'{data_root}/mow_{mode}.json', 'w') as json_data:
        coco_file = dict()
        data_images = []
        data_annos = []
        for idx in tqdm(range(len(meta_data))):
            meta = meta_data[idx]
            if meta['image_id'] not in data_list:
                continue

            img_info = dict()
            anno_info = dict()

            sample_id = idx
            img_info['id'] = sample_id
            img_info['file_name'] = meta['image_id']
            prefix = meta['image_id']
            
            anno_info['id'] = sample_id
            anno_info['image_id'] = sample_id
            img = cv2.imread(os.path.join(data_path, 'images', meta['image_id'] + '.jpg'))
            img_max = max(img.shape)
            anno_info['fx'] = img_max
            anno_info['fy'] = img_max
            anno_info['cx'] = 0.5 * img_max
            anno_info['cy'] = 0.5 * img_max

            mano_pose = torch.from_numpy(np.array(meta['hand_pose'], dtype=np.float32)).unsqueeze(0)
            mano_shape = torch.zeros((1, 10))
            th_trans = torch.from_numpy(np.array(meta['trans'], dtype=np.float32)).unsqueeze(0)
            hand_verts, hand_joints, full_hand_pose, global_trans, rot_center = mano_layer(mano_pose, th_betas=mano_shape, th_trans=th_trans, root_palm=False)
            hand_verts = hand_verts.squeeze().cpu().numpy()
            hand_joints = hand_joints.squeeze().cpu().numpy()
            hand_verts = hand_verts @ np.array(meta['hand_R']).reshape((3, 3))
            hand_joints = hand_joints @ np.array(meta['hand_R']).reshape((3, 3))
            hand_verts += np.array(meta['hand_t'])
            hand_joints += np.array(meta['hand_t'])
            hand_verts *= np.array(meta['hand_s'])
            hand_joints *= np.array(meta['hand_s'])
            hand_mesh = trimesh.Trimesh(vertices=hand_verts, faces=mano_faces)
            anno_info['hand_palm'] = ((np.array(hand_verts[95]) + np.array(hand_verts[22])) / 2).tolist()
            anno_info['hand_joints_3d'] = hand_joints.tolist()
            anno_info['hand_verts_3d'] = hand_verts.tolist()

            anno_info['obj_name'] = meta['obj_name']
            obj_mesh = trimesh.load(os.path.join(data_path, 'models', meta['image_id'] + '.obj'), process=False, force='mesh')
            obj_verts = obj_mesh.vertices - np.min(obj_mesh.vertices, axis=0)
            obj_verts /= np.abs(obj_verts).max()
            obj_verts *= 2
            obj_verts -= np.max(obj_verts, axis=0) / 2
            obj_verts -= obj_verts.mean(axis=0)
            obj_verts[:, 1] *= -1
            obj_faces = obj_mesh.faces[:, [2, 1, 0]]
            rest_obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)
            obj_verts = obj_verts @ np.array(meta['R']).reshape((3, 3))
            obj_verts += np.array(meta['t'])
            obj_verts *= np.array(meta['s'])
            obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)

            obj_transform = np.eye(4, dtype=np.float32)
            obj_transform[:3, :3] = np.array(meta['R']).reshape((3, 3)) * np.array(meta['s'])
            obj_transform[:3, 3] = np.array(meta['t']) * np.array(meta['s'])
            anno_info['obj_transform'] = obj_transform.tolist()
            anno_info['obj_center_3d'] = obj_transform[:3, 3].tolist()

            intr_mat = np.zeros((3, 4), dtype=np.float32)
            intr_mat[0, 0] = anno_info['fx']
            intr_mat[0, 2] = anno_info['cx']
            intr_mat[1, 1] = anno_info['fy']
            intr_mat[1, 2] = anno_info['cy']
            intr_mat[2, 2] = 1

            homo_obj_verts = np.ones((obj_verts.shape[0], 4), dtype=np.float32)
            homo_obj_verts[:, :3] = obj_verts
            obj_verts_2d = (intr_mat @ homo_obj_verts.transpose(1, 0)).transpose(1, 0)
            obj_verts_2d = obj_verts_2d[:, :2] / obj_verts_2d[:, [2]]

            homo_hand_verts = np.ones((hand_verts.shape[0], 4), dtype=np.float32)
            homo_hand_verts[:, :3] = hand_verts
            hand_verts_2d = (intr_mat @ homo_hand_verts.transpose(1, 0)).transpose(1, 0)
            hand_verts_2d = hand_verts_2d[:, :2] / hand_verts_2d[:, [2]]

            tl = np.min(np.concatenate([hand_verts_2d, obj_verts_2d], axis=0), axis=0)
            br = np.max(np.concatenate([hand_verts_2d, obj_verts_2d], axis=0), axis=0)
            box_size = br - tl
            bbox = np.concatenate([tl-10, box_size+20], axis=0)
            bbox = process_bbox(bbox)
            anno_info['bbox'] = bbox.tolist()

            # homo_hand_joints = np.ones((hand_joints.shape[0], 4), dtype=np.float32)
            # homo_hand_joints[:, :3] = hand_joints
            # hand_joints_2d = (intr_mat @ homo_hand_joints.transpose(1, 0)).transpose(1, 0)
            # hand_joints_2d = (hand_joints_2d[:, :2] / hand_joints_2d[:, [2]]).astype(np.int32)
            # for i in range(hand_joints_2d.shape[0]):
            #     cv2.circle(img, hand_joints_2d[i], 1, (0, 0, 255), 4)
            # cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 255, 0), 2)
            # cv2.imwrite('/home/zerchen/Bureau/test.jpg', img)
            # from IPython import embed; embed()

            hand_mesh.export(osp.join(mesh_hand_path, f'{prefix}.obj'))
            obj_mesh.export(osp.join(mesh_obj_path, f'{prefix}.obj'))
            rest_obj_mesh.export(osp.join(rest_mesh_obj_path, f'{prefix}.obj'))

            selected_ids.append(str(sample_id).rjust(8, '0'))
            data_images.append(img_info)
            data_annos.append(anno_info)

        coco_file['images'] = data_images
        coco_file['annotations'] = data_annos
        json.dump(coco_file, json_data, indent=2)

        with open(f'{data_root}/splits/{mode}.json', 'w') as f:
            json.dump(selected_ids, f, indent=2)

if __name__ == '__main__':
    Fire(preprocess)