#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :ho3dv3.py
#@Date        :2022/07/05 18:51:49
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import os
from turtle import color
import numpy as np
import time
import pickle
import json
import trimesh
import torch
import open3d as o3d
from tqdm import tqdm
from loguru import logger
from pycocotools.coco import COCO
from scipy.spatial import cKDTree as KDTree
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import iterative_closest_point
from utils.chamfer_utils import chamfer_distance


class hoi4d:
    def __init__(self, data_split, start_point=None, end_point=None):
        self.name = 'hoi4d'
        self.data_split = data_split
        self.start_point = start_point
        self.end_point = end_point

        self.stage_split = '_'.join(data_split.split('_')[:-1])
        split = self.stage_split.split('_')[-1]
        self.cur_dir = os.path.dirname(__file__)
        with open(os.path.join(self.cur_dir, 'splits', data_split + '.json'), 'r') as f:
            self.split = json.load(f)
        self.split = [int(idx) for idx in self.split]

        self.anno_file = os.path.join(self.cur_dir, 'data', f'{self.name}_{self.stage_split}.json')
        self.verts_hand_source = os.path.join(self.cur_dir, 'data', 'verts_3d')
        self.inria_aug_source = os.path.join(self.cur_dir, '..', 'inria_holidays', 'rgb')
        self.image_size = (1920, 1080)

        self.joints_name = ('wrist', 'index1', 'index2', 'index3', 'middle1', 'middle2', 'middle3', 'pinky1', 'pinky2', 'pinky3', 'ring1', 'ring2', 'ring3', 'thumb1', 'thumb2', 'thumb3', 'thumb4', 'index4', 'middle4', 'ring4', 'pinky4')
        self.data = self.load_data()

    def load_data(self):
        db = COCO(self.anno_file)
        data = []
        if self.start_point is None:
            self.start_point = 0

        if self.end_point is None:
            self.end_point = len(self.split)
        
        for aid in self.split[self.start_point:self.end_point]:
            sample = dict()
            ann = db.anns[aid]
            img_data = db.loadImgs(ann['image_id'])[0]

            sample['id'] = img_data['file_name']
            frame_id = sample['id'].split('-')[-1]
            subset = "/".join(sample['id'].split('-')[:-1])

            sample['img_path'] = os.path.join(self.cur_dir, 'data', "HOI4D_release", subset, "align_rgb", frame_id + ".jpg")
            if os.path.exists(os.path.join(self.cur_dir, 'data', "HOI4D_annotations", subset, "2Dseg", "mask", frame_id + '.png')):
                sample['seg_path'] = os.path.join(self.cur_dir, 'data', "HOI4D_annotations", subset, "2Dseg", "mask", frame_id + '.png')
            else:
                sample['seg_path'] = os.path.join(self.cur_dir, 'data', "HOI4D_annotations", subset, "2Dseg", "shift_mask", frame_id + '.png')
            sample['verts_hand_path'] = os.path.join(self.verts_hand_source, img_data['file_name'] + '.npy')

            sample['fx'] = ann['fx']
            sample['fy'] = ann['fy']
            sample['cx'] = ann['cx']
            sample['cy'] = ann['cy']

            sample['bbox'] = ann['bbox']
            sample['obj_name'] = ann['obj_name']
            obj_cat = sample['obj_name'].split("-")[0]
            obj_id = sample['obj_name'].split("-")[1]
            sample['mesh_obj_path'] = os.path.join(self.cur_dir, 'data', "HOI4D_CAD_Model_for_release", "rigid", obj_cat, f"{obj_id}.obj")
            sample['hand_side'] = np.array(1, dtype=np.float32)
            sample['hand_joints_3d'] = np.array(ann['hand_joints_3d'], dtype=np.float32)
            sample['obj_transform'] = np.array(ann['obj_transform'], dtype=np.float32)
            sample['obj_center_3d'] = np.array(ann['obj_transform'], dtype=np.float32)[:3, 3]
            sample['hand_palm'] = np.array(ann['hand_palm'], dtype=np.float32)
            sample['hand_poses'] = np.array(ann['hand_poses'], dtype=np.float32)
            sample['hand_shapes'] = np.array(ann['hand_shapes'], dtype=np.float32)
            sample['hand_trans'] = np.array(ann['hand_trans'], dtype=np.float32)
            if 'test' in self.stage_split:
                try:
                    sample['wilor_mano_hand_joints_3d'] = np.array(ann['wilor_mano_hand_joints_3d'], dtype=np.float32)
                    sample['wilor_mano_hand_verts_3d'] = np.array(ann['wilor_mano_hand_verts_3d'], dtype=np.float32)
                    sample['wilor_mano_hand_palm'] = np.array(ann['wilor_mano_hand_palm'], dtype=np.float32)
                    sample['wilor_pred_cam_t'] = np.array(ann['wilor_pred_cam_t'], dtype=np.float32)
                except:
                    pass

            data.append(sample)
        
        return data

    def _evaluate(self, output_path, idx):
        sample = self.data[idx]
        sample_idx = sample['id']

        pred_mano_pose_path = os.path.join(output_path, 'hand_pose', sample_idx + '.json')
        with open(pred_mano_pose_path, 'r') as f:
            pred_hand_pose = json.load(f)
        cam_extr = np.array(pred_hand_pose['cam_extr'])
        try:
            pred_mano_joint = (cam_extr @ np.array(pred_hand_pose['joints']).transpose(1, 0)).transpose(1, 0)
            if pred_mano_joint.shape[0] != sample['hand_joints_3d'].shape[0]:
                mano_joint_err = None
            else:
                mano_joint_err = np.mean(np.linalg.norm(pred_mano_joint - sample['hand_joints_3d'], axis=1)) * 1000.
        except:
            mano_joint_err = None

        pred_obj_pose_path = os.path.join(output_path, 'obj_pose', sample_idx + '.json')
        with open(pred_obj_pose_path, 'r') as f:
            pred_obj_pose = json.load(f)
        cam_extr = np.array(pred_obj_pose['cam_extr'])
        try:
            pred_obj_center = (cam_extr @ np.array(pred_obj_pose['center']).reshape((1, 3)).transpose(1, 0)).squeeze()
            obj_center_err = np.linalg.norm(pred_obj_center - sample['obj_center_3d']) * 1000.
        except:
            obj_center_err = None
        try:
            pred_obj_corners = (cam_extr @ np.array(pred_obj_pose['corners']).transpose(1, 0)).transpose(1, 0)
            obj_corner_err = np.mean(np.linalg.norm(pred_obj_corners - sample['obj_corners_3d'], axis=1)) * 1000.
        except:
            obj_corner_err = None

        pred_hand_mesh_path = os.path.join(output_path, 'sdf_mesh', sample_idx + '_hand.ply')
        gt_hand_mesh_path = os.path.join(self.cur_dir, 'data', 'mesh_data', 'mesh_hand', sample_idx + '.obj')
        if not os.path.exists(pred_hand_mesh_path):
            chamfer_hand = None
            fscore_hand_1 = None
            fscore_hand_5 = None
        else:
            pred_hand_mesh = trimesh.load(pred_hand_mesh_path, process=False)
            gt_hand_mesh = trimesh.load(gt_hand_mesh_path, process=False)

            pred_hand_points, _ = trimesh.sample.sample_surface(pred_hand_mesh, 30000)
            gt_hand_points, _ = trimesh.sample.sample_surface(gt_hand_mesh, 30000)
            pred_hand_points *= 100.
            gt_hand_points *= 100.

            # one direction
            gen_points_kd_tree = KDTree(pred_hand_points)
            one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_hand_points)
            gt_to_gen_chamfer = np.mean(np.square(one_distances))
            # other direction
            gt_points_kd_tree = KDTree(gt_hand_points)
            two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_hand_points)
            gen_to_gt_chamfer = np.mean(np.square(two_distances))
            chamfer_hand = gt_to_gen_chamfer + gen_to_gt_chamfer

            threshold = 0.1 # 1 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_hand_1 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

            threshold = 0.5 # 5 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_hand_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

        pred_obj_mesh_path = os.path.join(output_path, 'sdf_mesh', sample_idx + '_obj.ply')
        gt_obj_mesh_path = os.path.join(self.cur_dir, 'data', 'mesh_data', 'mesh_obj', sample_idx + '.obj')
        if not os.path.exists(pred_obj_mesh_path):
            chamfer_obj = None
            fscore_obj_5 = None
            fscore_obj_10 = None
        else:
            pred_obj_mesh = trimesh.load(pred_obj_mesh_path, process=False)
            gt_obj_mesh = trimesh.load(gt_obj_mesh_path, process=False)

            pred_obj_points, _ = trimesh.sample.sample_surface(pred_obj_mesh, 30000)
            gt_obj_points, _ = trimesh.sample.sample_surface(gt_obj_mesh, 30000)
            pred_obj_points *= 100.
            gt_obj_points *= 100.

            # one direction
            gen_points_kd_tree = KDTree(pred_obj_points)
            one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_obj_points)
            gt_to_gen_chamfer = np.mean(np.square(one_distances))
            # other direction
            gt_points_kd_tree = KDTree(gt_obj_points)
            two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_obj_points)
            gen_to_gt_chamfer = np.mean(np.square(two_distances))
            chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer

            threshold = 0.5 # 5 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

            threshold = 1.0 # 10 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
        
        error_dict = {}
        error_dict['id'] = sample_idx
        error_dict['chamfer_hand'] = chamfer_hand
        error_dict['fscore_hand_1'] = fscore_hand_1
        error_dict['fscore_hand_5'] = fscore_hand_5
        error_dict['chamfer_obj'] = chamfer_obj
        error_dict['fscore_obj_5'] = fscore_obj_5
        error_dict['fscore_obj_10'] = fscore_obj_10
        error_dict['mano_joint'] = mano_joint_err
        error_dict['obj_center'] = obj_center_err
        error_dict['obj_corner'] = obj_corner_err

        return error_dict

    def _evaluate_pc(self, output_path, idx, icp=False):
        sample = self.data[idx]
        sample_idx = sample['id']
        obj_name = sample['obj_name']

        pred_obj_pc_path = os.path.join(output_path, sample_idx + '.json')
        with open(pred_obj_pc_path, 'r') as f:
            json_data = json.load(f)
        cam_extr = np.array(json_data['cam_extr'], dtype=np.float32)

        if icp:
            pred_obj_points = (cam_extr @ np.array(json_data['pointclouds_up'], dtype=np.float32).transpose(1, 0)).transpose(1, 0)
            pred_obj_points = pred_obj_points + sample['obj_transform'][:3, 3]
            num_samples = 10000
            pred_pcd = o3d.geometry.PointCloud()
            pred_pcd.points = o3d.utility.Vector3dVector(pred_obj_points)
            cl, ind = pred_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
            pred_pcd = pred_pcd.select_by_index(ind)
            sampled_indices = np.random.choice(len(pred_pcd.points), size=10000, replace=False)
            pred_pcd = pred_pcd.select_by_index(sampled_indices)

            pred_obj_points = pred_obj_points[np.random.choice(pred_obj_points.shape[0], num_samples)]
            pred_obj_points = pred_obj_points.astype(np.float32)
            pred_pcd.points = o3d.utility.Vector3dVector(pred_obj_points)

            obj_cat = sample['obj_name'].split("-")[0]
            obj_id = sample['obj_name'].split("-")[1]
            gt_obj_mesh_path = os.path.join(self.cur_dir, 'data', "HOI4D_CAD_Model_for_release", "rigid", obj_cat, f"{obj_id}.obj")
            gt_obj_mesh = o3d.io.read_triangle_mesh(gt_obj_mesh_path)
            gt_obj_mesh = gt_obj_mesh.transform(sample['obj_transform'])
            gt_pcd = gt_obj_mesh.sample_points_uniformly(number_of_points=10000)
            gt_obj_center = sample['obj_transform'][:3, 3]

            icp_result = iterative_closest_point(torch.from_numpy(np.asarray(pred_pcd.points)[None]).float().cuda(), torch.from_numpy(np.asarray(gt_pcd.points)[None]).float().cuda(), init_transform=(torch.eye(3).unsqueeze(0).cuda(), torch.zeros(3).unsqueeze(0).cuda(), torch.ones(1).cuda()), max_iterations=50, relative_rmse_thr=1e-3, estimate_scale=False, verbose=False)
            pred_obj_points = icp_result[2][0].cpu().numpy()
            gt_obj_points = np.asarray(gt_pcd.points, dtype=np.float32)

            pred_obj_points *= 100.
            gt_obj_points *= 100.
            # one direction
            gen_points_kd_tree = KDTree(pred_obj_points)
            one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_obj_points)
            gt_to_gen_chamfer = np.mean(np.square(one_distances))
            # other direction
            gt_points_kd_tree = KDTree(gt_obj_points)
            two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_obj_points)
            gen_to_gt_chamfer = np.mean(np.square(two_distances))
            chamfer_obj = gt_to_gen_chamfer + gen_to_gt_chamfer

            threshold = 0.5 # 5 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_obj_5 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)

            threshold = 1.0 # 10 mm
            precision_1 = np.mean(one_distances < threshold).astype(np.float32)
            precision_2 = np.mean(two_distances < threshold).astype(np.float32)
            fscore_obj_10 = 2 * precision_1 * precision_2 / (precision_1 + precision_2 + 1e-7)
        else:
            pred_obj_points = (cam_extr @ np.array(json_data['pointclouds_up'], dtype=np.float32).transpose(1, 0)).transpose(1, 0) + cam_extr @ np.array(json_data['objtrans'], dtype=np.float32) + sample['hand_palm']
            num_samples = 30000
            pred_obj_points = pred_obj_points[np.random.choice(pred_obj_points.shape[0], num_samples)]
            pred_obj_points = pred_obj_points.astype(np.float32)

            obj_cat = sample['obj_name'].split("-")[0]
            obj_id = sample['obj_name'].split("-")[1]
            gt_obj_mesh_path = os.path.join(self.cur_dir, 'data', "HOI4D_CAD_Model_for_release", "rigid", obj_cat, f"{obj_id}.obj")
            gt_obj_mesh = trimesh.load(gt_obj_mesh_path, process=False)
            gt_obj_points, _ = trimesh.sample.sample_surface(gt_obj_mesh, num_samples)
            gt_obj_points = (sample['obj_transform'][:3, :3] @ gt_obj_points.transpose(1, 0)).transpose(1, 0) + sample['obj_transform'][:3, 3]
            gt_obj_points = gt_obj_points.astype(np.float32)
            gt_obj_center = sample['obj_transform'][:3, 3]

            (d10, d20), _ = chamfer_distance(torch.from_numpy(pred_obj_points).unsqueeze(0).cuda(), torch.from_numpy(gt_obj_points).unsqueeze(0).cuda(), batch_reduction=None, point_reduction=None)

            d1 = torch.sqrt(d10)
            d2 = torch.sqrt(d20)

            th_list = [5 / 1000., 10 / 1000.]
            res_list = []
            for th in th_list:
                if d1.size(1) and d2.size(1):
                    recall = torch.sum(d2 < th, dim=-1) / num_samples  # recall knn(gt, pred) gt->pred
                    precision = torch.sum(d1 < th, dim=-1) / num_samples  # precision knn(pred, gt) pred-->
                    eps = 1e-6
                    fscore = 2 * recall * precision / (recall + precision + eps)
                    res_list.append(fscore.tolist())
                else:
                    raise ValueError("d1 and d2 should be in equal length but got %d %d" % (d1.size(1), d2.size(1)))
            d = ((d10).mean(1) + (d20).mean(1)).tolist()

        pred_obj_center = cam_extr @ np.array(json_data['objtrans'], dtype=np.float32) + sample['hand_palm']
        obj_center_err = np.linalg.norm(pred_obj_center - gt_obj_center) * 1000
        
        error_dict = {}
        error_dict['id'] = sample_idx
        if icp:
            error_dict['chamfer_obj'] = chamfer_obj
            error_dict['fscore_obj_5'] = fscore_obj_5
            error_dict['fscore_obj_10'] = fscore_obj_10
        else:
            error_dict['chamfer_obj'] = d[0] * 1000
            error_dict['fscore_obj_5'] = res_list[0][0]
            error_dict['fscore_obj_10'] = res_list[1][0]
        error_dict['obj_center_err'] = obj_center_err

        return error_dict

    
if __name__ == "__main__":
    db = ho3dv3('test_s2_20k')
