#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File :vis.py
#@Date :2022/11/03 23:51:50
#@Author :zerui chen
#@Contact :zerui.chen@inria.fr

import argparse
import yaml
import json
import numpy as np
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import shutil
from multiprocessing import Process, Queue
import trimesh
from tqdm import tqdm
import cv2
import pyrender
import _init_paths

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-e', required=True, type=str)
    parser.add_argument('--id', '-i', required=True, type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    testset = args.dir.strip('/').split('/')[-1].split('_')[1]
    exec(f'from datasets.{testset}.{testset} import {testset}')

    with open(os.path.join(args.dir, '../exp.yaml'), 'r') as f:
        cfg = yaml.safe_load(f)

    if testset == 'obman':
        data_root = '../datasets/obman/data/'
        data_json = '../datasets/obman/obman_test.json'
        gt_mesh_hand_source = '../datasets/obman/data/test/mesh_hand/'
        gt_mesh_obj_source = '../datasets/obman/data/test/mesh_obj/'
    elif testset == 'dexycb':
        data_root = '../datasets/dexycb/data/'
        data_json = '../datasets/dexycb/dexycb_test_s0.json'
        from datasets.dexycb.toolkit.dex_ycb import _SUBJECTS
        gt_mesh_hand_source = '../datasets/dexycb/data/mesh_data/mesh_hand/'
        gt_mesh_obj_source = '../datasets/dexycb/data/mesh_data/mesh_obj/'

    output_vis_dir = os.path.join(args.dir, 'samples', args.id)
    os.makedirs(output_vis_dir, exist_ok=True)

    with open(data_json, 'r') as f:
        meta_data = json.load(f)
    
    idx_list = []
    fileanme_list = []
    for i in range(len(meta_data['images'])):
        if str(args.id) in str(meta_data['images'][i]['file_name']):
            idx_list.append(i)
            fileanme_list.append(meta_data['images'][i]['file_name'])
    
    for idx, sample_id in zip(idx_list, fileanme_list):
        if testset == 'obman':
            pass
        elif testset == 'dexycb':
            subject_id = _SUBJECTS[int(sample_id.split('_')[0]) - 1]
            video_id = '_'.join(sample_id.split('_')[1:3])
            cam_id = sample_id.split('_')[-2]
            frame_id = sample_id.split('_')[-1].rjust(6, '0')
            img_path = os.path.join(data_root, subject_id, video_id, cam_id, 'color_' + frame_id + '.jpg')
        
        pred_hand_mesh_path = os.path.join(args.dir, 'sdf_mesh', sample_id + '_hand.ply')
        pred_obj_mesh_path = os.path.join(args.dir, 'sdf_mesh', sample_id + '_obj.ply')
        
        os.system(f'cp {img_path} {output_vis_dir}')
        os.system(f'cp {pred_hand_mesh_path} {output_vis_dir}')
        os.system(f'cp {pred_obj_mesh_path} {output_vis_dir}')