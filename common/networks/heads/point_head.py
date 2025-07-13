#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :sdf_head.py
#@Date        :2022/04/09 16:57:10
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


class PointHead(nn.Module):
    def __init__(self, sdf_latent, point_latent, dims, dropout, dropout_prob, norm_layers, latent_in):
        super(PointHead, self).__init__()
        self.sdf_latent = sdf_latent
        self.point_latent = point_latent
        self.dims = [self.sdf_latent + self.point_latent] + dims + [3]
        self.num_layers = len(self.dims)
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        self.norm_layers = norm_layers
        self.latent_in = latent_in

        self.relu = nn.ReLU()

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in self.latent_in:
                out_dim = self.dims[layer + 1] - self.dims[0]
            else:
                out_dim = self.dims[layer + 1]

            if layer in self.norm_layers:
                setattr(self, "lin" + str(layer), nn.utils.weight_norm(nn.Linear(self.dims[layer], out_dim)),)
            else:
                setattr(self, "lin" + str(layer), nn.Linear(self.dims[layer], out_dim))

    def forward(self, input):
        latent = input
       
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                latent = torch.cat([latent, input], 2)
            latent = lin(latent)

            if layer < self.num_layers - 2:
                latent = self.relu(latent)
                if layer in self.dropout:
                    latent = F.dropout(latent, p=self.dropout_prob, training=self.training)

        return latent


if __name__ == "__main__":
    import sys
    net = SDFHead(256, 6, [512, 512, 512, 512], [0, 1, 2, 3], 0.2, [0, 1, 2, 3], [2], True, 6)
    input_size = (2, 262)
    input_tensor = torch.randn(input_size)
    latent, pred_class = net(input_tensor)
