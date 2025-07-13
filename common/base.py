import os
import os.path as osp
import math
import time
import glob
import abc
from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, MultiStepLR
import torch.optim
import torchvision.transforms as transforms
from utils.timer import Timer
from config import cfg
from net import get_model
from datasets.multiple_datasets import MultipleDatasets
from datasets.sdf_dataset import SDFDataset
from datasets.obj_sdf_dataset import OBJSDFDataset
from datasets.sdf_video_dataset import SDFVideoDataset
from datasets.obj_sdf_video_dataset import OBJSDFVideoDataset
from datasets.pose_dataset import PoseDataset
from datasets.pose_video_dataset import PoseVideoDataset
from datasets.point_cloud_dataset import PointCloudDataset


class Base(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, log_name ='logs.txt'):
        self.cur_epoch = 0
        # timer
        self.tot_timer = Timer()
        self.gpu_timer = Timer()
        self.read_timer = Timer()
        # logger
        self.logger = logger
        self.logger.add(osp.join(cfg.log_dir, log_name))

    @abc.abstractmethod
    def _make_batch_generator(self):
        return

    @abc.abstractmethod
    def _make_model(self):
        return


class Trainer(Base):
    def __init__(self):
        super(Trainer, self).__init__(log_name = 'train_logs.txt')

    def get_optimizer(self, model):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
        return optimizer

    def get_scheduler(self, optimizer):
        if cfg.warm_up_epoch > 0:
            warmup_scheduler = LinearLR(optimizer, total_iters=cfg.warm_up_epoch)
            if cfg.lr_dec_style == 'cosine':
                decay_scheduler = CosineAnnealingLR(optimizer, T_max=(cfg.end_epoch - cfg.warm_up_epoch), eta_min=1e-5)
            elif cfg.lr_dec_style == 'step':
                decay_scheduler = MultiStepLR(optimizer, milestones=cfg.lr_dec_epoch, gamma=cfg.lr_dec_factor)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[cfg.warm_up_epoch])
        else:
            if cfg.lr_dec_style == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=cfg.end_epoch, eta_min=1e-5)
            elif cfg.lr_dec_style == 'step':
                scheduler = MultiStepLR(optimizer, milestones=cfg.lr_dec_epoch, gamma=cfg.lr_dec_factor)

        return scheduler

    def save_model(self, state, epoch):
        file_path = osp.join(cfg.model_dir, 'snapshot_{}.pth.tar'.format(str(epoch)))
        torch.save(state, file_path)
        self.logger.info("Write snapshot into {}".format(file_path))

    def load_model(self, model, optimizer, scaler):
        model_file_list = glob.glob(osp.join(cfg.model_dir, '*.pth.tar'))
        if len(model_file_list) == 0:
            if os.path.exists(cfg.checkpoint):
                ckpt = torch.load(cfg.checkpoint, map_location=torch.device('cpu'))
                model.load_state_dict(ckpt['network'])
                start_epoch = 0
                self.logger.info('Load checkpoint from {}'.format(cfg.checkpoint))
                return start_epoch, model, optimizer, scaler
            else:
                start_epoch = 0
                self.logger.info('Start training from scratch')
                return start_epoch, model, optimizer, scaler
        else:
            cur_epoch = max([int(file_name[file_name.find('snapshot_') + 9 : file_name.find('.pth.tar')]) for file_name in model_file_list])
            ckpt_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu')) 
            start_epoch = ckpt['epoch'] + 1
            model.load_state_dict(ckpt['network'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scaler.load_state_dict(ckpt['scaler'])
            self.logger.info('Continue training and load checkpoint from {}'.format(ckpt_path))
            return start_epoch, model, optimizer, scaler

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']
    
    def _make_batch_generator(self):
        # data load and construct batch generator
        self.logger.info("Creating dataset...")
        # dynamic dataset import
        if '-' in cfg.trainset_3d:
            # currently only support osdf_video task
            train_datasets = cfg.trainset_3d.split('-')
            train_datasets_splits = cfg.trainset_3d_split.split('-')
            train_dataset_loaders = []
            for idx, _ in enumerate(train_datasets):
                exec(f'from datasets.{train_datasets[idx]}.{train_datasets[idx]} import {train_datasets[idx]}')
                trainset3d_db = eval(train_datasets[idx])('train_' + train_datasets_splits[idx])
                train_dataset_loaders.append(PointCloudDataset(trainset3d_db, cfg=cfg))

            self.trainset_loader = MultipleDatasets(train_dataset_loaders, make_same_len=False)
            self.itr_per_epoch = math.ceil(len(self.trainset_loader) / cfg.num_gpus / cfg.train_batch_size)
            self.train_sampler = DistributedSampler(self.trainset_loader)
            self.batch_generator = DataLoader(dataset=self.trainset_loader, batch_size=cfg.train_batch_size, shuffle=False, num_workers=cfg.num_threads, pin_memory=True, sampler=self.train_sampler, drop_last=False, persistent_workers=False)
        else:
            exec(f'from datasets.{cfg.trainset_3d}.{cfg.trainset_3d} import {cfg.trainset_3d}')
            if 'video' in cfg.task:
                trainset3d_db = eval(cfg.trainset_3d)('train_' + cfg.trainset_3d_split, video_mode=True, num_frames=cfg.num_frames)
            else:
                trainset3d_db = eval(cfg.trainset_3d)('train_' + cfg.trainset_3d_split)

            if cfg.task == 'hsdf_osdf_1net' or cfg.task == 'hsdf_osdf_2net' or cfg.task == 'hsdf_osdf_2net_pa' or cfg.task == 'pose_sdf' or cfg.task == 'hsdf_osdf_1net_plus' or cfg.task == 'hsdf_osdf_1net_plus_pa' or cfg.task == 'hsdf_osdf_3net_pa':
                self.trainset_loader = SDFDataset(trainset3d_db, cfg=cfg)
            elif cfg.task == 'hsdf_osdf_2net_video' or cfg.task == 'hsdf_osdf_2net_video_pa':
                self.trainset_loader = SDFVideoDataset(trainset3d_db, cfg=cfg)
            elif cfg.task == 'osdf_video':
                self.trainset_loader = OBJSDFVideoDataset(trainset3d_db, cfg=cfg)
            elif cfg.task == 'osdf_1net':
                self.trainset_loader = OBJSDFDataset(trainset3d_db, cfg=cfg)
            elif cfg.task == 'pose_kpt' or cfg.task == "obj_kpt":
                self.trainset_loader = PoseDataset(trainset3d_db, cfg=cfg)
            elif cfg.task == 'pose_kpt_video':
                self.trainset_loader = PoseVideoDataset(trainset3d_db, cfg=cfg)
            elif cfg.task == 'object_pc' or cfg.task == "object_pc_dino":
                self.trainset_loader = PointCloudDataset(trainset3d_db, cfg=cfg)

            self.itr_per_epoch = math.ceil(len(self.trainset_loader) / cfg.num_gpus / cfg.train_batch_size)
            self.train_sampler = DistributedSampler(self.trainset_loader)
            self.batch_generator = DataLoader(dataset=self.trainset_loader, batch_size=cfg.train_batch_size, shuffle=False, num_workers=cfg.num_threads, pin_memory=True, sampler=self.train_sampler, drop_last=False, persistent_workers=False)

    def _make_model(self, local_rank):
        # prepare network
        self.logger.info("Creating graph and optimizer...")
        model = get_model(cfg, True)
        model = model.cuda()
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        optimizer = self.get_optimizer(model)
        scaler = torch.amp.GradScaler()
        scheduler = self.get_scheduler(optimizer)
        model = NativeDDP(model, device_ids=[local_rank], output_device=local_rank)
        model.train()

        if (cfg.task == 'hsdf_osdf_2net' or cfg.task == 'hsdf_osdf_2net_pa' or cfg.task == 'pose_sdf' or cfg.task == 'hsdf_osdf_3net_pa') and os.path.exists(cfg.ckpt):
            ckpt = torch.load(cfg.ckpt, map_location=torch.device('cpu'))['network']
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            model.module.pose_model.load_state_dict(ckpt)
            self.logger.info('Load checkpoint from {}'.format(cfg.ckpt))
            model.module.pose_model.eval()
        elif (cfg.task == 'hsdf_osdf_2net_video' or cfg.task == 'hsdf_osdf_2net_video_pa') and os.path.exists(cfg.ckpt):
            ckpt = torch.load(cfg.ckpt, map_location=torch.device('cpu'))['network']
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            components = ['pose_model', 'backbone', 'neck', 'volume_head', 'hand_sdf_head', 'obj_sdf_head', 'backbone_2_sdf', 'sdf_encoder']
            components_ckpts = [{} for i in range(len(components))]
            for k, v in ckpt.items():
                for i in range(len(components)):
                    if k.split('.')[0] == components[i]:
                        new_key = '.'.join(k.split('.')[1:])
                        components_ckpts[i][new_key] = v
                
            model.module.pose_model.load_state_dict(components_ckpts[0])
            model.module.backbone.load_state_dict(components_ckpts[1])
            model.module.neck.load_state_dict(components_ckpts[2])
            model.module.volume_head.load_state_dict(components_ckpts[3])
            model.module.hand_sdf_head.load_state_dict(components_ckpts[4])
            model.module.obj_sdf_head.load_state_dict(components_ckpts[5])
            model.module.backbone_2_sdf.load_state_dict(components_ckpts[6])
            model.module.sdf_encoder.load_state_dict(components_ckpts[7])
            self.logger.info('Load checkpoint from {}'.format(cfg.ckpt))
            model.module.pose_model.eval()
        elif cfg.task == 'pose_kpt_video' and os.path.exists(cfg.ckpt):
            ckpt = torch.load(cfg.ckpt, map_location=torch.device('cpu'))['network']
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            components = ['backbone', 'neck', 'volume_head']
            components_ckpts = [{} for i in range(len(components))]
            for k, v in ckpt.items():
                for i in range(len(components)):
                    if k.split('.')[0] == components[i]:
                        new_key = '.'.join(k.split('.')[1:])
                        components_ckpts[i][new_key] = v
            model.module.backbone.load_state_dict(components_ckpts[0])
            model.module.neck.load_state_dict(components_ckpts[1])
            model.module.volume_head.load_state_dict(components_ckpts[2])
            self.logger.info('Load checkpoint from {}'.format(cfg.ckpt))
        elif (cfg.task == 'hsdf_osdf_1net_plus' or cfg.task == 'hsdf_osdf_1net_plus_pa') and os.path.exists(cfg.ckpt):
            ckpt = torch.load(cfg.ckpt, map_location=torch.device('cpu'))['network']
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            components = ['backbone', 'neck', 'volume_head']
            components_ckpts = [{} for i in range(len(components))]
            for k, v in ckpt.items():
                for i in range(len(components)):
                    if k.split('.')[0] == components[i]:
                        new_key = '.'.join(k.split('.')[1:])
                        components_ckpts[i][new_key] = v
            model.module.backbone.load_state_dict(components_ckpts[0])
            model.module.neck_hand.load_state_dict(components_ckpts[1])
            model.module.volume_head_hand.load_state_dict(components_ckpts[2])
            self.logger.info('Load checkpoint from {}'.format(cfg.ckpt))
        elif cfg.task == 'osdf_video' and os.path.exists(cfg.ckpt):
            ckpt = torch.load(cfg.ckpt, map_location=torch.device('cpu'))['network']
            ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
            components = ['backbone', 'sdf_head', 'backbone_2_sdf', 'sdf_encoder']
            if cfg.seg_branch:
                components += ['neck']
            components_ckpts = [{} for i in range(len(components))]
            for k, v in ckpt.items():
                for i in range(len(components)):
                    if k.split('.')[0] == components[i]:
                        new_key = '.'.join(k.split('.')[1:])
                        components_ckpts[i][new_key] = v
            model.module.backbone.load_state_dict(components_ckpts[0])
            model.module.sdf_head.load_state_dict(components_ckpts[1])
            model.module.backbone_2_sdf.load_state_dict(components_ckpts[2])
            model.module.sdf_encoder.load_state_dict(components_ckpts[3])
            if cfg.seg_branch:
                model.module.neck.load_state_dict(components_ckpts[4])
            self.logger.info('Load checkpoint from {}'.format(cfg.ckpt))
        elif cfg.task == 'object_pc_dino' and os.path.exists(cfg.ckpt):
            if 'model_lvis_rel.ckpt' in cfg.ckpt:
                ckpt = torch.load(cfg.ckpt, map_location=torch.device('cpu'))['state_dict']
                components = ['tokenizer', 'backbone', 'post_processor']
                components_ckpts = [{} for i in range(len(components))]
                for k, v in ckpt.items():
                    for i in range(len(components)):
                        if k.split('.')[1] == components[i] and k.split('.')[0] == "pointcloud_generator":
                            new_key = '.'.join(k.split('.')[2:])
                            components_ckpts[i][new_key] = v
                model.module.tokenizer.load_state_dict(components_ckpts[0])
                model.module.backbone.load_state_dict(components_ckpts[1])
                model.module.post_processor.load_state_dict(components_ckpts[2])
                self.logger.info('Load checkpoint from {}'.format(cfg.ckpt))
            else:
                ckpt = torch.load(cfg.ckpt, map_location=torch.device('cpu'))['network']
                ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
                components = ['image_tokenizer', 'tokenizer', 'backbone', 'post_processor', 'post_processor_trans', 'pointnet', 'pointcloud_upsampler']
                components_ckpts = [{} for i in range(len(components))]
                for k, v in ckpt.items():
                    for i in range(len(components)):
                        if k.split('.')[0] == components[i]:
                            new_key = '.'.join(k.split('.')[1:])
                            components_ckpts[i][new_key] = v
                model.module.image_tokenizer.load_state_dict(components_ckpts[0])
                model.module.tokenizer.load_state_dict(components_ckpts[1])
                model.module.backbone.load_state_dict(components_ckpts[2])
                model.module.post_processor.load_state_dict(components_ckpts[3])
                model.module.post_processor_trans.load_state_dict(components_ckpts[4])
                model.module.pointnet.load_state_dict(components_ckpts[5])
                model.module.pointcloud_upsampler.load_state_dict(components_ckpts[6])
                self.logger.info('Load checkpoint from {}'.format(cfg.ckpt))

        start_epoch, model, optimizer, scaler = self.load_model(model, optimizer, scaler)

        self.start_epoch = start_epoch
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler


class Tester(Base):
    def __init__(self, local_rank, test_epoch):
        self.local_rank = local_rank
        self.test_epoch = test_epoch
        super(Tester, self).__init__(log_name = 'test_logs.txt')

    def _make_batch_generator(self):
        # data load and construct batch generator
        start_points = []
        end_points = []
        if 'video' in cfg.task:
            tot_test_samples = cfg.num_testset_videos
        else:
            tot_test_samples = cfg.num_testset_samples

        division = tot_test_samples // cfg.num_gpus
        for i in range(cfg.num_gpus):
            start_point = i * division
            if i != cfg.num_gpus - 1:
                end_point = start_point + division
            else:
                end_point = tot_test_samples
            start_points.append(start_point)
            end_points.append(end_point)

        self.logger.info(f"Creating dataset from {start_points[self.local_rank]} to {end_points[self.local_rank]}")
        exec(f'from datasets.{cfg.testset}.{cfg.testset} import {cfg.testset}')
        if 'video' in cfg.task:
            testset3d_db = eval(cfg.testset)('test_' + cfg.testset_split, start_points[self.local_rank], end_points[self.local_rank], video_mode=True, num_frames=cfg.num_frames, use_whole_video_test=cfg.use_whole_video_test)
        else:
            testset3d_db = eval(cfg.testset)('test_' + cfg.testset_split, start_points[self.local_rank], end_points[self.local_rank])

        if cfg.task == 'hsdf_osdf_1net' or cfg.task == 'hsdf_osdf_2net' or cfg.task == 'hsdf_osdf_2net_pa' or cfg.task == 'pose_sdf' or cfg.task == 'hsdf_osdf_1net_plus' or cfg.task == 'hsdf_osdf_1net_plus_pa' or cfg.task == 'hsdf_osdf_3net_pa':
            self.testset_loader = SDFDataset(testset3d_db, cfg=cfg, mode='test')
        elif cfg.task == 'hsdf_osdf_2net_video' or cfg.task == 'hsdf_osdf_2net_video_pa':
            self.testset_loader = SDFVideoDataset(testset3d_db, cfg=cfg, mode='test')
        elif cfg.task == 'osdf_video':
            self.testset_loader = OBJSDFVideoDataset(testset3d_db, cfg=cfg, mode='test')
        elif cfg.task == 'osdf_1net':
            self.testset_loader = OBJSDFDataset(testset3d_db, cfg=cfg, mode='test')
        elif cfg.task == 'pose_kpt' or cfg.task == "obj_kpt":
            self.testset_loader = PoseDataset(testset3d_db, cfg=cfg, mode='test')
        elif cfg.task == 'pose_kpt_video':
            self.testset_loader = PoseVideoDataset(testset3d_db, cfg=cfg, mode='test')
        elif cfg.task == 'object_pc' or cfg.task == "object_pc_dino":
            self.testset_loader = PointCloudDataset(testset3d_db, cfg=cfg, mode='test')

        self.itr_per_epoch = math.ceil(len(self.testset_loader) / cfg.num_gpus / cfg.test_batch_size)
        self.batch_generator = DataLoader(dataset=self.testset_loader, batch_size=cfg.test_batch_size, shuffle=False, num_workers=cfg.num_threads, pin_memory=True, drop_last=False, persistent_workers=False)
    
    def _make_model(self, local_rank):
        model_path = os.path.join(cfg.model_dir, 'snapshot_%d.pth.tar' % self.test_epoch)
        assert os.path.exists(model_path), 'Cannot find model at ' + model_path
        self.logger.info('Load checkpoint from {}'.format(model_path))
        
        # prepare network
        self.logger.info("Creating graph...")
        model = get_model(cfg, is_train=False)
        model = model.cuda()
        model = NativeDDP(model, device_ids=[local_rank], output_device=local_rank)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['network'])
        model.eval()

        self.model = model
