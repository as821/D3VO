

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


class Networks():
    def __init__(self, encoder_path="encoder.pth", decoder_path="depth.pth"):
        """Initialize DepthNet and PoseNet from pretrained weights"""
        # Initialize DepthNet from pretrained weights
        model_name = "mono_640x192"
        download_model_if_doesnt_exist(model_name)
        encoder_path = os.path.join("models", model_name, encoder_path)
        depth_decoder_path = os.path.join("models", model_name, decoder_path)

        # LOADING PRETRAINED MODEL
        self.encoder = networks.ResnetEncoder(18, False)
        self.depth_decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(loaded_dict)

        self.encoder.eval()
        self.depth_decoder.eval()

        self.h = loaded_dict_enc['height']
        self.w = loaded_dict_enc['width']


        # TODO initialize PoseNet

        

    def depth(self, img, visualize=False):
        """Get a depth prediction from DepthNet for the given image"""
        #image_path = "/Users/andrewstange/Desktop/CMU/Fall_2022/16-833/Project/D3VO/monodepth2/assets/test_image.jpg"

        #input_image = pil.open(image_path).convert('RGB')
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

        # Return as numpy array (self.width, self.heights)
        return disp_resized_np.T

