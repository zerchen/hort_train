#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :cocoify_obman.py
#@Date        :2022/04/20 16:18:30
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

from distutils.log import debug
import numpy as np
import torch
import os
import os.path as osp
from tqdm import tqdm
from fire import Fire
import json
import pickle
import shutil
import trimesh
from scipy.spatial import cKDTree as KDTree
import cv2
import sys
from glob import glob
import cv2
import lmdb
sys.path.insert(0, '../common')
from mano.manolayer import ManoLayer
from utils.img_utils import generate_patch_image, process_bbox


def convert_pose_to_opencv(pose, trans):
    coordChangMat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    newRot = cv2.Rodrigues(coordChangMat.dot(cv2.Rodrigues(pose[:3])[0]))[0][:,0]
    new_trans = trans.copy().dot(coordChangMat.T)
    new_pose = pose.copy()
    new_pose[:3] = newRot

    return new_pose, new_trans

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)

    return img_coord


def preprocess(data_root='../datasets/ho3dv3', mode='test', split='s1'):
    # mode can be train or test
    # s0 is the standard split with sdf training
    # s1 is my split of the training set for mesh evaluation
    with open(os.path.join(data_root, 'data', mode + '_s1.json'), 'r') as f:
        data_list = json.load(f)

    with open('/home/zerchen/workspace/dataset/ho3dv3/evaluation_verts.json', 'r') as f:
        verts_data = json.load(f)

    with open('/home/zerchen/workspace/dataset/ho3dv3/evaluation_xyz.json', 'r') as f:
        joints_data = json.load(f)

    with open(os.path.join(data_root, 'data', 'evaluation.txt'), 'r') as f:
        index_list = f.readlines()
    index_list = ['evaluation-' + filename.strip().replace('/', '-') for filename in index_list]

    verts_root_path = f'{data_root}/verts_3d'
    os.makedirs(verts_root_path, exist_ok=True)
    
    selected_ids = []
    with open(f'{data_root}/ho3dv3_{mode}_{split}_t.json', 'w') as json_data:
        coco_file = dict()
        data_images = []
        data_annos = []
        sample_id = 0

        for idx in tqdm(range(len(data_list))):
            subset = data_list[idx].split('-')[0]
            video_id = data_list[idx].split('-')[1]
            frame_idx = data_list[idx].split('-')[2]

            img_info = dict()
            anno_info = dict()
            img_info['id'] = sample_id
            img_info['file_name'] = data_list[idx]
            anno_info['id'] = sample_id
            anno_info['image_id'] = sample_id
            meta_path = os.path.join(data_root, 'data', subset, video_id, 'meta', frame_idx + '.pkl')
            with open(meta_path, 'rb') as f:
                meta_data = pickle.load(f)

            anno_info['fx'] = float(meta_data['camMat'][0, 0])
            anno_info['cx'] = float(meta_data['camMat'][0, 2])
            anno_info['fy'] = float(meta_data['camMat'][1, 1])
            anno_info['cy'] = float(meta_data['camMat'][1, 2])

            # img_path = os.path.join(data_root, 'data', subset, video_id, 'rgb', frame_idx + '.jpg')
            # img = cv2.imread(img_path)

            if subset == 'train':
                hand_joints_3d = meta_data['handJoints3D'][[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20], :]
                hand_joints_3d[:, 1] *= -1
                hand_joints_3d[:, 2] *= -1
                anno_info['hand_joints_3d'] = hand_joints_3d.tolist()

                mano_layer = ManoLayer(flat_hand_mean=True, side="right", mano_root='../common/mano/assets/', use_pca=False)
                hand_verts_3d, _, _, _, _ = mano_layer(torch.from_numpy(meta_data['handPose'].astype(np.float32)).unsqueeze(0), torch.from_numpy(meta_data['handBeta'].astype(np.float32)).unsqueeze(0), torch.from_numpy(meta_data['handTrans'].astype(np.float32)).unsqueeze(0))
                hand_verts_3d = hand_verts_3d.squeeze().cpu().numpy()
                hand_verts_3d[:, 1] *= -1
                hand_verts_3d[:, 2] *= -1
                anno_info['hand_palm'] = ((hand_verts_3d[95] + hand_verts_3d[22]) / 2).tolist()
            else:
                hand_joints_3d = np.array(joints_data[index_list.index(data_list[idx])], dtype=np.float32)[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20], :]
                hand_joints_3d[:, 1] *= -1
                hand_joints_3d[:, 2] *= -1
                anno_info['hand_joints_3d'] = hand_joints_3d.tolist()

                hand_verts_3d = np.array(verts_data[index_list.index(data_list[idx])], dtype=np.float32)
                hand_verts_3d[:, 1] *= -1
                hand_verts_3d[:, 2] *= -1
                anno_info['hand_palm'] = ((hand_verts_3d[95] + hand_verts_3d[22]) / 2).tolist()
            
            if mode == "train":
                np.save(os.path.join(verts_root_path, f'{data_list[idx]}.npy'), hand_verts_3d.astype(np.float32))

            anno_info['obj_name'] = meta_data['objName']
            obj_rot, obj_trans = convert_pose_to_opencv(meta_data['objRot'].squeeze(), meta_data['objTrans'])

            obj_corners_3d = meta_data['objCorners3D'].copy()
            obj_corners_3d[:, 1] *= -1
            obj_corners_3d[:, 2] *= -1
            anno_info['obj_corners_3d'] = obj_corners_3d.tolist()
            anno_info['obj_center_3d'] = obj_trans.tolist()

            obj_rest_corners_3d = meta_data['objCorners3DRest'].copy()
            anno_info['obj_rest_corners_3d'] = obj_rest_corners_3d.tolist()

            obj_transform = np.zeros((4, 4), dtype=np.float32)
            obj_transform[3, 3] = 1
            obj_transform[:3, :3] = cv2.Rodrigues(obj_rot)[0]
            obj_transform[:3, 3] = obj_trans
            anno_info['obj_transform'] = obj_transform.tolist()

            obj_img = cam2pixel(obj_corners_3d, f=[anno_info['fx'], anno_info['fy']], c=[anno_info['cx'], anno_info['cy']])[:, :2]
            hand_img = cam2pixel(hand_joints_3d, f=[anno_info['fx'], anno_info['fy']], c=[anno_info['cx'], anno_info['cy']])[:, :2]
            tl = np.min(np.concatenate([hand_img, obj_img], axis=0), axis=0)
            br = np.max(np.concatenate([hand_img, obj_img], axis=0), axis=0)
            box_size = br - tl
            bbox = np.concatenate([tl-10, box_size+20], axis=0)
            bbox = process_bbox(bbox)
            anno_info['bbox'] = bbox.tolist()

            selected_ids.append(str(sample_id).rjust(8, '0'))
            data_images.append(img_info)
            data_annos.append(anno_info)

            sample_id += 1

        coco_file['images'] = data_images
        coco_file['annotations'] = data_annos
        json.dump(coco_file, json_data, indent=2)

        with open(f'{data_root}/splits/{mode}_{split}.json', 'w') as f:
            json.dump(selected_ids, f, indent=2)


if __name__ == '__main__':
    Fire(preprocess)
