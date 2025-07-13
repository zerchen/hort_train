#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
#@File        :unet.py
#@Date        :2022/04/10 11:55:30
#@Author      :zerui chen
#@Contact     :zerui.chen@inria.fr

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, inplanes, outplanes, num_stages, use_final_layers=False):
        super(UNet, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.num_stages = num_stages
        self.use_final_layers = use_final_layers
        self.deconv_layers = self._make_deconv_layer(self.num_stages)
        if self.use_final_layers:
            final_layers = []
            for i in range(3):
                final_layers.append(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
                final_layers.append(nn.BatchNorm2d(self.outplanes))
                final_layers.append(nn.ReLU(inplace=True))
            self.final_layers = nn.Sequential(*final_layers)

    def _make_deconv_layer(self, num_stages):
        layers = []
        for i in range(num_stages):
            layers.append(nn.ConvTranspose2d(in_channels=self.inplanes, out_channels=self.outplanes, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False))
            layers.append(nn.BatchNorm2d(self.outplanes))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = self.outplanes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.deconv_layers(x)
        if self.use_final_layers:
            x = self.final_layers(x)
        return x


class UNetV2(nn.Module):
    def __init__(self):
        super(UNetV2, self).__init__()
        conv_layers = []
        conv_layers.append(nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1, padding=1))
        conv_layers.append(nn.BatchNorm2d(256))
        conv_layers.append(nn.ReLU(inplace=True))
        self.conv_layers = nn.Sequential(*conv_layers)

        self.deconv_layer_1 = self._make_deconv_layer(1)
        self.deconv_layer_2 = self._make_deconv_layer(1)

    def _make_deconv_layer(self, num_stages):
        layers = []
        for i in range(num_stages):
            layers.append(nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, output_padding=0, bias=False))
            layers.append(nn.BatchNorm2d(256))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv_layers(x)
        x2 = self.deconv_layer_1(x1)
        x3 = self.deconv_layer_2(x2)
        return [x1, x2, x3]
