# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict


class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=None, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.a_conv = nn.Conv2d(256, num_frames_to_predict_for, 1)
        self.b_conv = nn.Conv2d(256, num_frames_to_predict_for, 1)

        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        out_ab = None
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

            if i==1:
                out_ab = out

        out_pose = out.mean(3).mean(2)


        out_a = self.a_conv(out_ab)
        out_a = self.softplus(out_a)
        out_a = out_a.mean(3).mean(2)
        out_b = self.b_conv(out_ab)
        out_b = self.tanh(out_b)
        out_b = out_b.mean(3).mean(2)

        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)
        out_a = 0.01 * out_a.view(-1, self.num_frames_to_predict_for, 1, 1)
        out_b = 0.01 * out_b.view(-1, self.num_frames_to_predict_for, 1, 1)

        axisangle = out_pose[..., :3]
        translation = out_pose[..., 3:]
        a = out_a
        b = out_b

        return axisangle, translation, a, b
