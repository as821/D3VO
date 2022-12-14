

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import sys

import torch
from torchvision import transforms

# include monodepth2 directory in module search
sys.path.insert(1, os.path.join(sys.path[0], '../monodepth2'))
import networks
from utils import download_model_if_doesnt_exist
from layers import disp_to_depth, transformation_from_parameters


# Set heuristically
MAX_DEPTH = 300
MIN_DEPTH = 0.2


class Networks():
    def __init__(self, weights_dir):
        """Initialize DepthNet and PoseNet from pretrained weights"""
        depthencoder_path=os.path.join(weights_dir, "encoder.pth")
        depth_decoder_path=os.path.join(weights_dir, "depth.pth")
        pose_path=os.path.join(weights_dir, "pose.pth")

        # Initialize DepthNet from pretrained weights
        self.encoder = networks.ResnetEncoder(18, False)
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict_enc = torch.load(depthencoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(loaded_dict)

        self.encoder.eval()
        self.depth_decoder.eval()

        self.h = loaded_dict_enc['height']
        self.w = loaded_dict_enc['width']

        # Initialize PoseNet
        self.posenet = networks.PoseCNN(num_input_frames=2)
        self.posenet.load_state_dict(torch.load(pose_path, map_location="cpu"))
        self.posenet.eval()
        

    def depth(self, img, visualize=False):
        """Get a depth prediction from DepthNet for the given image"""
        original_width, original_height = img.shape[1], img.shape[0]
        feed_height = self.h
        feed_width = self.w
        input_img = img
        img = pil.fromarray(img)
        input_image_resized = img.resize((feed_width, feed_height), pil.LANCZOS)
        input_image_pytorch = transforms.ToTensor()(input_image_resized).unsqueeze(0)

        with torch.no_grad():
            features = self.encoder(input_image_pytorch)
            outputs = self.depth_decoder(features)
        disp_resized = torch.nn.functional.interpolate(outputs[("disp", 0)],(original_height, original_width), mode="bilinear", align_corners=False)
        disp_resized_np = disp_resized.squeeze().cpu().numpy()

        _, pred_depth = disp_to_depth(outputs[("disp", 0)], MIN_DEPTH, MAX_DEPTH)
        depth_resized = torch.nn.functional.interpolate(pred_depth,(original_height, original_width), mode="bilinear", align_corners=False).numpy().squeeze()
        sigma_resized = torch.nn.functional.interpolate(outputs[("disp-sigma", 0)], (original_height, original_width), mode="bilinear", align_corners=False).numpy().squeeze()

        if visualize:
            vmax = np.percentile(disp_resized_np, 95)
            plt.figure(figsize=(10, 10))
            plt.subplot(211)
            plt.imshow(input_img)
            plt.title("Input", fontsize=22)
            plt.axis('off')

            plt.subplot(212)
            plt.imshow(disp_resized_np, cmap='magma', vmax=vmax)
            plt.title("Disparity prediction", fontsize=22)
            plt.axis('off')

            mx = np.argmax(depth_resized.T)
            mx_coord = (int(mx / depth_resized.shape[0]), int(mx % depth_resized.shape[0]))
            mn = np.argmin(depth_resized.T)
            mn_coord = (int(mn / depth_resized.shape[0]), int(mn % depth_resized.shape[0]))
            print("Max depth: ", np.max(depth_resized), mx_coord, ", min depth: ", np.min(depth_resized), mn_coord)

            plt.plot(mx_coord[0], mx_coord[1], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")
            plt.plot(mn_coord[0], mn_coord[1], marker="o", markersize=10, markeredgecolor="red", markerfacecolor="red")

        # Return as numpy array (self.width, self.heights)
        return depth_resized, sigma_resized


    def pose(self, img1, img2, depth, translation_scale=45):
        """Pass provided image pair through PoseNet. Returns pose estimate (4x4 homogenous matrix) from img1 to img2."""
        # Resize images to fit the pose network
        assert img1.shape == img2.shape
        img1 = pil.fromarray(img1)
        img1 = transforms.ToTensor()(img1.resize((self.w, self.h), pil.LANCZOS)).unsqueeze(0)
        img2 = pil.fromarray(img2)
        img2 = transforms.ToTensor()(img2.resize((self.w, self.h), pil.LANCZOS)).unsqueeze(0)
        with torch.no_grad():
            axisangle, translation, a, b = self.posenet(torch.cat((img1, img2), 1))
        return transformation_from_parameters(axisangle[:, 0], translation[:, 0] * (1 / depth).mean() * translation_scale).cpu().numpy().squeeze(), a.numpy(), b.numpy()
