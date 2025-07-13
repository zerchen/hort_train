import torch
import torch.nn as nn
import numpy as np
import time
import tgs
from torch.nn import functional as F
from pytorch3d.loss import chamfer_distance
# from emd import earth_mover_distance
# from sdf.sdf_loss import SDFLoss, sdf_extractor
from networks.backbones.pointnet import PointNetEncoder
from networks.backbones.volume_extractor import VolumeEncoder
from networks.tgs.models.snowflake.model_spdpp import mask_generation
from networks.tgs.models.snowflake.pointnet2 import PointNet2SemSegSSG
from networks.heads.point_head import PointHead
from networks.necks.unet import UNetV2
from networks.lora.encoder_lora import DINOV2EncoderLoRA
from networks.tgs.models.snowflake.pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
from utils.sdf_utils import point_align


class model(nn.Module):
    def __init__(self, cfg):
        super(model, self).__init__()
        self.cfg = cfg
        self.mano_faces = np.load('../common/mano/assets/closed_fmano.npy')

        self.image_tokenizer = tgs.find(self.cfg.image_tokenizer_cls)(self.cfg.image_tokenizer)
        if self.cfg.use_lora:
            self.image_tokenizer = DINOV2EncoderLoRA(self.image_tokenizer)
        else:
            if self.cfg.image_tokenizer_name == "dinov2-base":
                for name, param in self.image_tokenizer.named_parameters():
                    param.requires_grad = False
                    if ('layer.6' in name) or ('layer.7' in name) or ('layer.8' in name) or ('layer.9' in name) or ('layer.10' in name) or ('layer.11' in name):
                        param.requires_grad = True
            elif self.cfg.image_tokenizer_name == "dinov2-large":
                for name, param in self.image_tokenizer.named_parameters():
                    param.requires_grad = False
                    if ('layer.12' in name) or ('layer.13' in name) or ('layer.14' in name) or ('layer.15' in name) or ('layer.16' in name) or ('layer.17' in name) or ('layer.18' in name) or ('layer.19' in name) or ('layer.20' in name) or ('layer.21' in name) or ('layer.22' in name) or ('layer.23' in name):
                        param.requires_grad = True

        pointnet_output_channels = 768 if self.cfg.image_tokenizer_name == "dinov2-base" else 1024
        if self.cfg.hand_kine:
            self.pointnet = PointNetEncoder(67, pointnet_output_channels)
        else:
            self.pointnet = PointNetEncoder(4, pointnet_output_channels)

        self.tokenizer = tgs.find(self.cfg.tokenizer_cls)(self.cfg.tokenizer)
        self.backbone = tgs.find(self.cfg.backbone_cls)(self.cfg.backbone)
        self.post_processor = tgs.find(self.cfg.post_processor_cls)(self.cfg.post_processor)
        self.post_processor_trans = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
        self.pointcloud_upsampler = tgs.find(self.cfg.pointcloud_upsampler_cls)(self.cfg.pointcloud_upsampler)
        if self.cfg.use_sdf:
            self.sdf_extractor = sdf_extractor(self.mano_faces)
            self.volume_encoder = VolumeEncoder(pointnet_output_channels)

        self.chamfer_loss = chamfer_distance
        # self.earth_mover_loss = earth_mover_distance
        self.l1_loss = nn.L1Loss()

    #cond_input may include camera intrinsics or hand wrist position
    def forward(self, inputs, targets=None, metas=None, mode='train'):
        if mode == 'train':
            input_img = inputs['img']
            batch_size = input_img.shape[0]

            encoder_hidden_states = self.image_tokenizer(input_img, None) # B * C * Nt
            encoder_hidden_states = encoder_hidden_states.transpose(2, 1) # B * Nt * C

            palm_norm_hand_verts_3d = metas['right_hand_verts_3d'] - metas['right_hand_palm'].unsqueeze(1)
            point_idx = torch.arange(778).view(1, 778, 1).expand(batch_size, -1, -1).to(input_img.device) / 778.
            palm_norm_hand_verts_3d = torch.cat([palm_norm_hand_verts_3d, point_idx], -1)
            if self.cfg.hand_kine:
                tip_norm_hand_verts_3d = (metas['right_hand_verts_3d'].unsqueeze(2) - metas['right_hand_joints_3d'].unsqueeze(1)).reshape((batch_size, 778, -1))
                norm_hand_verts_3d = torch.cat([palm_norm_hand_verts_3d, tip_norm_hand_verts_3d], -1)
                hand_feats = self.pointnet(norm_hand_verts_3d)
            else:
                hand_feats = self.pointnet(palm_norm_hand_verts_3d)

            tokens = self.tokenizer(batch_size)
            if self.cfg.use_sdf:
                sdf_feats = self.sdf_extractor(palm_norm_hand_verts_3d)
                sdf_feats = self.volume_encoder(sdf_feats.unsqueeze(1)).reshape((batch_size, 4, -1))
                tokens = self.backbone(tokens, torch.cat([encoder_hidden_states, hand_feats.unsqueeze(1), sdf_feats], 1), modulation_cond=None)
            else:
                tokens = self.backbone(tokens, torch.cat([encoder_hidden_states, hand_feats.unsqueeze(1)], 1), modulation_cond=None)
            tokens = self.tokenizer.detokenize(tokens)

            pointclouds = self.post_processor(tokens[:, :2048, :])
            pred_obj_trans = self.post_processor_trans(tokens[:, -1, :])

            upsampling_input = {
                "input_image_tokens": encoder_hidden_states.permute(0, 2, 1),
                "intrinsic_cond": metas['cam_intr'],
                "points": pointclouds,
                "hand_points": metas["right_hand_verts_3d"],
                "trans": pred_obj_trans + metas['right_hand_palm'],
                "scale": self.cfg.recon_scale
            }
            up_results = self.pointcloud_upsampler(upsampling_input)
            pointclouds_up = up_results[-1]
            # pred_mask = mask_generation(pointclouds_up * self.cfg.recon_scale + (pred_obj_trans + metas['right_hand_palm']).unsqueeze(1), metas['cam_intr'][:, :3, :3], input_img)

            pc_results = {}
            pc_results['pointclouds'] = pointclouds
            pc_results['objtrans'] = pred_obj_trans
            pc_results['pointclouds_up'] = pointclouds_up

            tgt_pointclouds = targets['point_clouds']
            tgt_obj_trans = metas['obj_transform'][:, :3, 3] - metas['right_hand_palm']
            tgt_pointclouds_up = targets['point_clouds_up']
            # tgt_mask = targets['obj_seg']

            loss = {}
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=False):
                loss_chamfer, _ = self.chamfer_loss(pointclouds.float(), tgt_pointclouds.float())
                # loss_earth_mover = torch.mean(self.earth_mover_loss(pointclouds.float(), tgt_pointclouds.float(), transpose=False) / 2048)
                loss_obj_trans = self.l1_loss(pred_obj_trans.float(), tgt_obj_trans.float())
                loss_chamfer_up, _ = self.chamfer_loss(pointclouds_up.float(), tgt_pointclouds_up.float())
            # loss_mask = self.l1_loss(pred_mask, tgt_mask)

            loss['chamfer'] = loss_chamfer * 20
            # loss['earth_mover'] = loss_earth_mover * 10
            loss['obj_trans'] = loss_obj_trans * 20
            loss['chamfer_up'] = loss_chamfer_up * 10
            # loss['mask'] = loss_mask * 1

            return loss, pc_results
        else:
            with torch.no_grad():
                input_img = inputs['img']
                batch_size = input_img.shape[0]

                encoder_hidden_states = self.image_tokenizer(input_img, None) # B * C * Nt
                encoder_hidden_states = encoder_hidden_states.transpose(2, 1) # B * Nt * C

                palm_norm_hand_verts_3d = metas['right_hand_verts_3d'] - metas['right_hand_palm'].unsqueeze(1)
                point_idx = torch.arange(778).view(1, 778, 1).expand(batch_size, -1, -1).to(input_img.device) / 778.
                palm_norm_hand_verts_3d = torch.cat([palm_norm_hand_verts_3d, point_idx], -1)
                if self.cfg.hand_kine:
                    tip_norm_hand_verts_3d = (metas['right_hand_verts_3d'].unsqueeze(2) - metas['right_hand_joints_3d'].unsqueeze(1)).reshape((batch_size, 778, -1))
                    norm_hand_verts_3d = torch.cat([palm_norm_hand_verts_3d, tip_norm_hand_verts_3d], -1)
                    hand_feats = self.pointnet(norm_hand_verts_3d)
                else:
                    hand_feats = self.pointnet(palm_norm_hand_verts_3d)

                tokens = self.tokenizer(batch_size)
                if self.cfg.use_sdf:
                    sdf_feats = self.sdf_extractor(palm_norm_hand_verts_3d)
                    sdf_feats = self.volume_encoder(sdf_feats.unsqueeze(1)).reshape((batch_size, 4, -1))
                    tokens = self.backbone(tokens, torch.cat([encoder_hidden_states, hand_feats.unsqueeze(1), sdf_feats], 1), modulation_cond=None)
                else:
                    tokens = self.backbone(tokens, torch.cat([encoder_hidden_states, hand_feats.unsqueeze(1)], 1), modulation_cond=None)
                tokens = self.tokenizer.detokenize(tokens)

                pointclouds = self.post_processor(tokens[:, :2048, :])
                pred_obj_trans = self.post_processor_trans(tokens[:, -1, :])

                upsampling_input = {
                    "input_image_tokens": encoder_hidden_states.permute(0, 2, 1),
                    "intrinsic_cond": metas['cam_intr'],
                    "points": pointclouds,
                    "hand_points": metas["right_hand_verts_3d"],
                    "trans": pred_obj_trans + metas['right_hand_palm'],
                    "scale": self.cfg.recon_scale
                }
                up_results = self.pointcloud_upsampler(upsampling_input)
                pointclouds_up = up_results[-1]

                pc_results = {}
                pc_results['pointclouds'] = pointclouds
                pc_results['objtrans'] = pred_obj_trans
                pc_results['handpalm'] = metas['right_hand_palm']
                pc_results['pointclouds_up'] = pointclouds_up

            return pc_results


def get_model(cfg, is_train):
    ho_model = model(cfg)

    return ho_model


if __name__ == '__main__':
    model = get_model(cfg, True)
    input_size = (2, 3, 256, 256)
    input_img = torch.randn(input_size)
    input_point_size = (2, 2000, 3)
    input_points = torch.randn(input_point_size)
    sdf_results, hand_pose_results, obj_pose_results = model(input_img, input_points)
