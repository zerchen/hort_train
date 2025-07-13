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


def get_color_map(N=256):
    """
    Return the color (R, G, B) of each label index.
    """
    
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap


def parse_2Dmask_img(mask_img, N=10):
    """
    mask_img: RGB image, shape = (H, W, 3)
    N: number of labels (including background)

    return: pixel labels, shape = (H, W)
    """

    color_map = get_color_map(N=N)

    H, W = mask_img.shape[:2]
    labels = np.zeros((H, W)).astype(np.uint8)

    for i in range(N):
        c = color_map[i]
        valid = (mask_img[..., 0] == c[0]) & (mask_img[..., 1] == c[1]) & (mask_img[..., 2] == c[2])
        labels[valid] = i
    
    return labels


def connected_mask(mask1, mask2):
    """
    Determine if two masks are connected in the image.

    Args:
        mask1 (np.ndarray): Binary mask for the first region (1920x1080).
        mask2 (np.ndarray): Binary mask for the second region (1920x1080).

    Returns:
        bool: True if the masks are connected, False otherwise.
    """
    # Ensure inputs are binary masks
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    # Dilate both masks to ensure connectivity is captured
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    mask1_dilated = cv2.dilate(mask1, kernel, iterations=1)
    mask2_dilated = cv2.dilate(mask2, kernel, iterations=1)

    # Check for overlap between the dilated masks
    overlap = mask1_dilated & mask2_dilated

    return np.any(overlap)


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


def preprocess(data_root='../datasets/hoi4d', mode='train', split='s1'):
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
    
    selected_ids = []
    with open(f'{data_root}/hoi4d_{mode}_{split}_t.json', 'w') as json_data:
        right_hand_dir = os.path.join(data_root, 'data', 'handpose', 'refinehandpose_right')
        left_hand_dir = os.path.join(data_root, 'data', 'handpose', 'refinehandpose_left')
        rgb_dir = os.path.join(data_root, 'data', 'HOI4D_release')
        anno_dir = os.path.join(data_root, 'data', 'HOI4D_annotations')
        object_dir = os.path.join(data_root, 'data', 'HOI4D_CAD_Model_for_release', 'rigid')
        output_verts_3d_dir = os.path.join(data_root, 'data', 'verts_3d')
        os.makedirs(output_verts_3d_dir, exist_ok=True)
        object_mapping = ['', 'ToyCar', 'Mug', 'Laptop', 'StorageFurniture', 'Bottle', 'Safe', 'Bowl', 'Bucket', 'Scissors', '', 'Pliers', 'Kettle', 'Knife', 'TrashCan', '', '', 'Lamp', 'Stapler', '', 'Chair']

        init_rgb_path_list = sorted(glob(f"{rgb_dir}/*/*/*/*/*/*/*/align_rgb/*.jpg"))
        
        right_rgb_path_list = []
        right_obj_path_list = []
        right_hand_path_list = []
        left_rgb_path_list = []
        left_obj_path_list = []
        left_hand_path_list = []
        for idx, path in enumerate(init_rgb_path_list):
            suffix_ori = path.split('/')[-1].split('.')[0]
            suffix_abb = str(int(path.split('/')[-1].split('.')[0]))
            
            obj_path_dir = '/'.join(path.replace(rgb_dir, anno_dir).replace('align_rgb', 'objpose').split('/')[:-1])
            obj_category_name = object_mapping[int(obj_path_dir.split('/')[7].split('C')[-1])]
            if obj_category_name not in ["Bottle", "Bowl", "Kettle", "Knife", "Mug", "ToyCar"]:
                continue
            if os.path.exists(os.path.join(obj_path_dir, f"{suffix_ori}.json")):
                obj_path = os.path.join(obj_path_dir, f"{suffix_ori}.json")
            elif os.path.exists(os.path.join(obj_path_dir, f"{suffix_abb}.json")):
                obj_path = os.path.join(obj_path_dir, f"{suffix_abb}.json")
            else:
                continue
            
            right_hand_path_dir = '/'.join(path.replace(rgb_dir, right_hand_dir).replace('align_rgb/', '').split('/')[:-1])
            if os.path.exists(os.path.join(right_hand_path_dir, f"{suffix_ori}.pickle")):
                right_hand_path = os.path.join(right_hand_path_dir, f"{suffix_ori}.pickle")
            elif os.path.exists(os.path.join(right_hand_path_dir, f"{suffix_abb}.pickle")):
                right_hand_path = os.path.join(right_hand_path_dir, f"{suffix_abb}.pickle")
            else:
                right_hand_path = None

            left_hand_path_dir = '/'.join(path.replace(rgb_dir, left_hand_dir).replace('align_rgb/', '').split('/')[:-1])
            if os.path.exists(os.path.join(left_hand_path_dir, f"{suffix_ori}.pickle")):
                left_hand_path = os.path.join(left_hand_path_dir, f"{suffix_ori}.pickle")
            elif os.path.exists(os.path.join(left_hand_path_dir, f"{suffix_abb}.pickle")):
                left_hand_path = os.path.join(left_hand_path_dir, f"{suffix_abb}.pickle")
            else:
                left_hand_path = None

            if right_hand_path is not None and left_hand_path is None:
                right_rgb_path_list.append(path)
                right_hand_path_list.append(right_hand_path)
                right_obj_path_list.append(obj_path)

            if right_hand_path is None and left_hand_path is not None:
                left_rgb_path_list.append(path)
                left_hand_path_list.append(left_hand_path)
                left_obj_path_list.append(obj_path)

        coco_file = dict()
        data_images = []
        data_annos = []
        sample_id = 0
        for idx, (rgb_path, hand_path, obj_path) in tqdm(enumerate(zip(right_rgb_path_list, right_hand_path_list, right_obj_path_list))):
            img_info = dict()
            anno_info = dict()
            
            frame_idx = hand_path.split('/')[-1].split('.')[0].rjust(5, '0')
            seq_name = '-'.join(hand_path.split('/')[6:-1])
            sample_name = '-'.join(hand_path.split('/')[6:-1] + [frame_idx])
            obj_category_name = object_mapping[int(hand_path.split('/')[8].split('C')[-1])]
            obj_instance_name = hand_path.split('/')[9].split('N')[-1].rjust(3, '0')
            obj_name = f"{obj_category_name}-{obj_instance_name}"

            if mode == "train":
                if seq_name in (val_seq_list + test_seq_list):
                    continue
            elif mode == "test":
                if seq_name not in test_seq_list:
                    continue

            seg_path_dir = '/'.join(obj_path.split('/')[:-1]).replace('objpose', '2Dseg')
            if os.path.exists(os.path.join(seg_path_dir, 'mask', f'{frame_idx}.png')):
                seg_path = os.path.join(seg_path_dir, 'mask', f'{frame_idx}.png')
            elif os.path.exists(os.path.join(seg_path_dir, 'shift_mask', f'{frame_idx}.png')):
                seg_path = os.path.join(seg_path_dir, 'shift_mask', f'{frame_idx}.png')
            else:
                continue

            if int(frame_idx) % 5 != 0 and mode == "test":
                continue

            ho_seg = cv2.imread(seg_path)
            ho_seg = parse_2Dmask_img(ho_seg)
            hand_seg = (ho_seg == 2)
            obj_seg = (ho_seg == 4)
            connected = connected_mask(hand_seg, obj_seg)
            if not connected:
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
            np.save(os.path.join(output_verts_3d_dir, f"{sample_name}.npy"), hand_verts_3d)

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