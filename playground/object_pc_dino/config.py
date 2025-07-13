#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :config.py
#@Date        :2022/04/08 09:47:39
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import os
import os.path as osp
import sys
from yacs.config import CfgNode as CN
from loguru import logger
from contextlib import redirect_stdout


cfg = CN()

cfg.task = 'object_pc_dino'
cfg.cur_dir = osp.dirname(os.path.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, '..', '..')
cfg.data_dir = osp.join(cfg.root_dir, 'datasets')
cfg.output_dir = '.'
cfg.model_dir = './model_dump'
cfg.vis_dir = './vis'
cfg.log_dir = './log'
cfg.result_dir = './result'
cfg.ckpt = '.'

## dataset
cfg.trainset_3d = 'h2o3d'
cfg.trainset_3d_split = '61k'
cfg.testset = 'h2o3d'
cfg.testset_split = '15k'
cfg.testset_hand_source = osp.join(cfg.testset, 'data/test/mesh_hand')
cfg.testset_obj_source = osp.join(cfg.testset, 'data/test/mesh_obj')
cfg.num_testset_samples = 15342
cfg.chamfer_optim = True

## model setting
cfg.hand_branch = True
cfg.obj_branch = True
cfg.hand_kine = True

cfg.image_tokenizer_name = "dinov2-large"
cfg.image_tokenizer_cls = "tgs.models.tokenizers.image.DINOV2SingleImageTokenizer"
cfg.image_tokenizer = CN()
cfg.image_tokenizer.pretrained_model_name_or_path = f"facebook/{cfg.image_tokenizer_name}"
cfg.image_tokenizer.width = 224
cfg.image_tokenizer.height = 224
cfg.image_tokenizer.modulation = False
cfg.image_tokenizer.modulation_zero_init = True
cfg.image_tokenizer.modulation_cond_dim = 768 if cfg.image_tokenizer_name == "dinov2-base" else 1024
cfg.image_tokenizer.freeze_backbone_params = False
cfg.image_tokenizer.enable_memory_efficient_attention = False
cfg.image_tokenizer.enable_gradient_checkpointing = False

cfg.tokenizer_cls = "tgs.models.tokenizers.point.PointLearnablePositionalEmbedding"
cfg.tokenizer = CN()
cfg.tokenizer.num_pcl = 2049
cfg.tokenizer.num_channels = 512

cfg.backbone_cls = "tgs.models.transformers.Transformer1D"
cfg.backbone = CN()
cfg.backbone.in_channels = cfg.tokenizer.num_channels
cfg.backbone.num_attention_heads = 8
cfg.backbone.attention_head_dim = 64
cfg.backbone.num_layers = 10
cfg.backbone.cross_attention_dim = 768 if cfg.image_tokenizer_name == "dinov2-base" else 1024
cfg.backbone.norm_type = "layer_norm"
cfg.backbone.enable_memory_efficient_attention = False
cfg.backbone.gradient_checkpointing = False

cfg.post_processor_cls = "tgs.models.networks.PointOutLayer"
cfg.post_processor = CN()
cfg.post_processor.in_channels = 512
cfg.post_processor.out_channels = 3

cfg.pointcloud_upsampler_cls = "tgs.models.snowflake.model_spdpp.SnowflakeModelSPDPP"
cfg.pointcloud_upsampler = CN()
cfg.pointcloud_upsampler.input_channels = 768 if cfg.image_tokenizer_name == "dinov2-base" else 1024
cfg.pointcloud_upsampler.dim_feat = 128
cfg.pointcloud_upsampler.num_p0 = 2048
cfg.pointcloud_upsampler.radius = 1
cfg.pointcloud_upsampler.bounding = True
cfg.pointcloud_upsampler.use_fps = True
cfg.pointcloud_upsampler.up_factors = [2, 4]
cfg.pointcloud_upsampler.token_type = "image_token"

## training config
cfg.image_size = (224, 224)
cfg.warm_up_epoch = 0
cfg.lr_dec_epoch = [40, 80]
cfg.end_epoch = 120
cfg.lr = 1e-4
cfg.lr_dec_style = 'step'
cfg.lr_dec_factor = 0.5
cfg.train_batch_size = 64
cfg.num_sample_points = 2048
cfg.recon_scale = 0.3
cfg.use_sdf = False
cfg.use_seg = False
cfg.use_lora = False

## testing config
cfg.test_batch_size = 1
cfg.test_with_gt_hand = False

## others
cfg.use_lmdb = False
cfg.num_threads = 6
cfg.gpu_ids = (0, 1, 2, 3)
cfg.num_gpus = 4
cfg.checkpoint = 'model.pth.tar'
cfg.model_save_freq = 10
cfg.use_inria_aug = True
cfg.use_3d_aug = False

def update_config(cfg, args, mode='train'):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.gpu_ids = args.gpu_ids
    cfg.num_gpus = len(cfg.gpu_ids.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    logger.info('>>> Using GPU: {}'.format(cfg.gpu_ids))

    if mode == 'train':
        exp_info = [cfg.image_tokenizer_name, cfg.trainset_3d, cfg.trainset_3d_split, 'e' + str(cfg.end_epoch), 'b' + str(cfg.num_gpus * cfg.train_batch_size), 'img' + str(cfg.image_size[0]), 'h' + str(int(cfg.hand_branch)), 'hkine' + str(int(cfg.hand_kine)), 'bgaug' + str(int(cfg.use_inria_aug)), '3daug' + str(int(cfg.use_3d_aug)), 'lr' + str(cfg.lr), 'sdf' + str(int(cfg.use_sdf)), 'seg' + str(int(cfg.use_seg)), "lora" + str(int(cfg.use_lora))]

        cfg.output_dir = osp.join(cfg.root_dir, 'outputs', cfg.task, '_'.join(exp_info))
        cfg.model_dir = osp.join(cfg.output_dir, 'model_dump')
        cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
        cfg.log_dir = osp.join(cfg.output_dir, 'log')
        cfg.result_dir = osp.join(cfg.output_dir, '_'.join(['result', cfg.testset, 'gthand' + str(int(cfg.test_with_gt_hand))]))

        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.model_dir, exist_ok=True)
        os.makedirs(cfg.vis_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.result_dir, exist_ok=True)
        cfg.freeze()
        with open(osp.join(cfg.output_dir, 'exp.yaml'), 'w') as f:
            with redirect_stdout(f): print(cfg.dump())
    else:
        cfg.result_dir = osp.join(cfg.output_dir, '_'.join(['result', cfg.testset, 'gthand' + str(int(cfg.test_with_gt_hand))]))
        cfg.log_dir = osp.join(cfg.output_dir, 'test_log')
        os.makedirs(cfg.result_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        cfg.freeze()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
sys.path.insert(0, osp.join(cfg.root_dir, 'common', 'networks'))
sys.path.insert(0, osp.join(cfg.root_dir, 'common', 'emd'))
from utils.dir_utils import add_pypath
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
