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
from scipy.spatial.transform import Rotation as Rot
import cv2
import sys
from glob import glob
import cv2
import lmdb
from glob import glob
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


def preprocess(data_root='../datasets/hoi4d', mode='test', split='s0'):
    # mode can be train or test
    # s0 is the standard split with sdf training
    # s1 is my split of the training set for mesh evaluation

    with open(os.path.join(data_root, 'test.txt'), 'r') as f:
        test_seq_list = f.readlines()
    test_seq_list = [filename.strip().replace('_', '-') for filename in test_seq_list]

    with open(os.path.join(data_root, 'val.txt'), 'r') as f:
        val_seq_list = f.readlines()
    val_seq_list = [filename.strip().replace('_', '-') for filename in val_seq_list]

    mano_layer = ManoLayer(flat_hand_mean=True, side="right", mano_root='../common/mano/assets/', use_pca=False)
    mano_layer = mano_layer.cuda()
    mano_faces = np.load("../common/mano/assets/closed_fmano.npy")
    
    selected_ids = []
    with open(f'{data_root}/hoi4d_{mode}_{split}_t.json', 'w') as json_data:
        right_hand_dir = os.path.join(data_root, 'data', 'handpose', 'refinehandpose_right')
        rgb_dir = os.path.join(data_root, 'data', 'HOI4D_release')
        anno_dir = os.path.join(data_root, 'data', 'HOI4D_annotations')
        object_dir = os.path.join(data_root, 'data', 'HOI4D_CAD_Model_for_release', 'rigid')
        output_verts_3d_dir = os.path.join(data_root, 'data', 'verts_3d')
        os.makedirs(output_verts_3d_dir, exist_ok=True)
        object_mapping = ['', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle', 'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle', 'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair']

        init_hand_path_list = sorted(glob(f"{right_hand_dir}/*/*/*/*/*/*/*/*.pickle"))
        hand_path_list = []
        obj_path_list = []
        rgb_path_list = []
        for idx, path in enumerate(init_hand_path_list):
            obj_path = path.replace(right_hand_dir, anno_dir)
            rgb_path = path.replace(right_hand_dir, rgb_dir)
            obj_path = os.path.join('/'.join(obj_path.split('/')[:-1]), 'objpose', obj_path.split('/')[-1].replace('pickle', 'json'))
            rgb_path = os.path.join('/'.join(rgb_path.split('/')[:-1]), 'align_rgb', rgb_path.split('/')[-1].split('.')[0].rjust(5, '0') + '.jpg')
            if os.path.exists(rgb_path) and os.path.exists(obj_path):
                hand_path_list.append(path)
                obj_path_list.append(obj_path)
                rgb_path_list.append(rgb_path)
        
        coco_file = dict()
        data_images = []
        data_annos = []
        sample_id = 0
        for idx, (rgb_path, hand_path, obj_path) in tqdm(enumerate(zip(rgb_path_list, hand_path_list, obj_path_list))):
            img_info = dict()
            anno_info = dict()
            
            frame_idx = hand_path.split('/')[-1].split('.')[0].rjust(5, '0')
            seq_name = '-'.join(hand_path.split('/')[6:-1])
            sample_name = '-'.join(hand_path.split('/')[6:-1] + [frame_idx])
            obj_category_name = object_mapping[int(hand_path.split('/')[8].split('C')[-1])]
            obj_instance_name = hand_path.split('/')[9].split('N')[-1].rjust(3, '0')
            obj_name = f"{obj_category_name}-{obj_instance_name}"
            if obj_category_name not in ["Bottle", "Bowl", "Kettle", "Knife", "Mug", "ToyCar"]:
                continue

            if int(frame_idx) % 5 != 0:
                continue

            if mode == "train":
                if seq_name in (val_seq_list + test_seq_list):
                    continue
            elif mode == "test":
                if seq_name not in test_seq_list:
                    continue

            cam_name = hand_path.split('/')[6]
            cam_intr = np.load(os.path.join(data_root, 'data', 'camera_params', cam_name, 'intrin.npy'))

            img_info['id'] = idx
            img_info['file_name'] = sample_name
            anno_info['id'] = idx
            anno_info['image_id'] = idx

            anno_info['fx'] = cam_intr[0, 0]
            anno_info['cx'] = cam_intr[0, 2]
            anno_info['fy'] = cam_intr[1, 1]
            anno_info['cy'] = cam_intr[1, 2]

            with open(hand_path, 'rb') as f:
                hand_anno = pickle.load(f)
            pose_param = torch.from_numpy(hand_anno['poseCoeff'][None]).cuda()
            shape_param = torch.from_numpy(hand_anno['beta'][None]).cuda()
            trans_param = torch.from_numpy(hand_anno['trans'][None]).cuda()
            hand_verts_3d, hand_joints_3d, _, _, _ = mano_layer(pose_param, shape_param, trans_param)
            hand_verts_3d = hand_verts_3d.squeeze(0).detach().cpu().numpy()
            hand_joints_3d = hand_joints_3d.squeeze(0).detach().cpu().numpy()

            obj_mesh = trimesh.load(os.path.join(object_dir, obj_category_name, f"{obj_instance_name}.obj"), process=False)
            obj_pose = np.eye(4, dtype=np.float32)
            with open(obj_path, 'r') as f:
                obj_anno = json.load(f)
            trans = obj_anno['dataList'][0]['center']
            trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)
            obj_pose[:3, 3] = trans.astype(np.float32)
            rot = obj_anno['dataList'][0]['rotation']
            rot = np.array([rot['x'], rot['y'], rot['z']])
            rot = Rot.from_euler('XYZ', rot).as_matrix()
            obj_pose[:3, :3] = rot.astype(np.float32)
            obj_rest_verts = np.asarray(obj_mesh.vertices, dtype=np.float32)
            obj_verts = (obj_pose[:3, :3] @ obj_rest_verts.transpose(1, 0)).transpose(1, 0) + obj_pose[:3, 3]

            hand_points_kd_tree = KDTree(hand_verts_3d)
            obj2hand_distances, _ = hand_points_kd_tree.query(obj_verts)
            if obj2hand_distances.min() > 0.005:
                continue

            np.save(os.path.join(output_verts_3d_dir, f"{sample_name}.npy"), hand_verts_3d)

            anno_info['hand_joints_3d'] = hand_joints_3d.tolist()
            anno_info['hand_palm'] = ((hand_verts_3d[95] + hand_verts_3d[22]) / 2).tolist()
            anno_info['hand_poses'] = hand_anno['poseCoeff'].tolist()
            anno_info['hand_shapes'] = hand_anno['beta'].tolist()
            anno_info['hand_trans'] = hand_anno['trans'].tolist()

            anno_info['obj_name'] = obj_name
            anno_info['obj_center_3d'] = obj_pose[:3, 3].tolist()
            anno_info['obj_transform'] = obj_pose.tolist()

            obj_img = cam2pixel(obj_verts, f=[anno_info['fx'], anno_info['fy']], c=[anno_info['cx'], anno_info['cy']])[:, :2]
            hand_img = hand_anno['kps2D'].astype(np.float32)
            tl = np.min(np.concatenate([hand_img, obj_img], axis=0), axis=0)
            br = np.max(np.concatenate([hand_img, obj_img], axis=0), axis=0)
            box_size = br - tl
            bbox = np.concatenate([tl-10, box_size+20], axis=0)
            bbox = process_bbox(bbox)
            anno_info['bbox'] = bbox.tolist()

            # intr_mat = np.zeros((3, 4), dtype=np.float32)
            # intr_mat[0, 0] = anno_info['fx']
            # intr_mat[0, 2] = anno_info['cx']
            # intr_mat[1, 1] = anno_info['fy']
            # intr_mat[1, 2] = anno_info['cy']
            # intr_mat[2, 2] = 1
            # img = cv2.imread(rgb_path)
            # homo_obj_verts = np.ones((hand_joints_3d.shape[0], 4), dtype=np.float32)
            # homo_obj_verts[:, :3] = hand_joints_3d
            # verts_2d = (intr_mat @ homo_obj_verts.transpose(1, 0)).transpose(1, 0)
            # verts_2d = (verts_2d[:, :2] / verts_2d[:, [2]]).astype(np.int32)
            # for i in range(verts_2d.shape[0]):
            #     if verts_2d[i, 0] >= 0 and verts_2d[i, 0] <= 1920 and verts_2d[i, 1] >= 0 and verts_2d[i, 1] <= 1080:
            #         cv2.circle(img, verts_2d[i], 1, (0, 0, 255), 4)
            # cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (255, 0, 0), 4)
            # cv2.imwrite('/home/zerchen/Bureau/test.jpg', img)
            # from IPython import embed; embed()

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
    # Fire(create_lmdb)