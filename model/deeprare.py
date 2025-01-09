import numpy as np
from numpy import expand_dims

import torch
import torch.nn.functional as F
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms as transforms
import torchvision.models as models

# Define a function to get the outputs from all layers
class IntermediateLayerGetter(nn.Module):
    def __init__(self, model, layer_names):
        super(IntermediateLayerGetter, self).__init__()
        self.model = model
        self.layer_names = layer_names
        self.outputs = []

    def forward(self, x):
        self.outputs = []
        for name, layer in self.model.features._modules.items():
            x = layer(x)
            if name in self.layer_names:
                self.outputs.append(x)
        return self.outputs
    
class RarityNetwork(nn.Module):

    def __init__(self, threshold= None):
        super(RarityNetwork, self).__init__()
        self.threshold= threshold
    

    def rarity_tensor(self, channel):
        B,C , a , b = channel.shape
        if a > 50:  # manage margins for low-level features
            channel[:,:,0:3, :] = 0
            channel[:,:,:, a - 3:a] = 0
            channel[:,:,:, 0:3] = 0
            channel[:,:,b - 3:b, :] = 0
        if a == 28:  # manage margins for mid-level features
            channel[:,:,0:2, :] = 0
            channel[:,:,:, a - 2:a] = 0
            channel[:,:,:, 0:2] = 0
            channel[:,:,b - 2:b, :] = 0
        if a == 14:  # manage margins for high-level features
            channel[:,:,0:1, :] = 0
            channel[:,:,:, a - 1:a] = 0
            channel[:,:,:, 0:1] = 0
            channel[:,:,b - 1:b, :] = 0

        
        channel = channel.view(B,C, -1)
        tensor_min = channel.min(dim=2, keepdim=True)[0]
        tensor_max = channel.max(dim=2, keepdim=True)[0]
        channel = (channel - tensor_min) / (tensor_max - tensor_min + 1e-8)
        channel = channel * 256

        bins = 11

        min_val, max_val = channel.min(), channel.max()
        bin_edges = torch.linspace(min_val, max_val, steps=bins + 1, device=channel.device)

        bin_indices = torch.bucketize(channel, bin_edges, right=True) - 1
        bin_indices = bin_indices.clamp(0, bins - 1)

        histograms = torch.zeros((B,C, bins), device=channel.device)
        histograms.scatter_add_(2, bin_indices, torch.ones_like(bin_indices, dtype=torch.float, device=channel.device))    

        histograms = histograms / histograms.sum(dim=2, keepdim=True)
        histograms = -torch.log(histograms + 1e-4)
        hists_idx = ((channel/256.) * (bins - 1)).long().clamp(0, bins - 1)

        dst = histograms.gather(2, hists_idx)

        tensor_min = dst.min(dim=2, keepdim=True)[0]
        tensor_max = dst.max(dim=2, keepdim=True)[0]
        dst = (dst - tensor_min) / (tensor_max - tensor_min + 1e-8)

        if self.threshold is not None:
            dst[dst < self.threshold] = 0

        map_max = dst.max(dim=2, keepdim=True)[0]
        map_mean = dst.mean(dim=2, keepdim=True)
        map_weight = (map_max - map_mean) ** 2  # Itti-like weight
        dst = dst * map_weight

        dst = torch.pow(dst, 2)
        ma = dst.max(dim=2, keepdim=True)[0]
        me = dst.mean(dim=2, keepdim=True)
        w = (ma - me) * (ma - me)
        dst = w * dst

        tensor_min = dst.min(dim=2, keepdim=True)[0]
        tensor_max = dst.max(dim=2, keepdim=True)[0]
        dst = (dst - tensor_min) / (tensor_max - tensor_min + 1e-8)
        dst = dst.view(B,C, a,b)

        if a > 50:
            dst[:,:,0:3, :] = 0
            dst[:,:,:, a - 3:a] = 0
            dst[:,:,:, 0:3] = 0
            dst[:,:,b - 3:b, :] = 0
        if a == 28:
            dst[:,:,0:2, :] = 0
            dst[:,:,:, a - 2:a] = 0
            dst[:,:,:, 0:2] = 0
            dst[:,:,b - 2:b, :] = 0
        if a == 14:
            dst[:,:,0:2, :] = 0
            dst[:,:,:, a - 2:a] = 0
            dst[:,:,:, 0:2] = 0
            dst[:,:,b - 2:b, :] = 0
        if a < 14:
            dst[:,:,0:2, :] = 0
            dst[:,:,:, a - 2:a] = 0
            dst[:,:,:, 0:2] = 0
            dst[:,:,b - 2:b, :] = 0

        dst = dst.view(B,C, -1)
        dst = torch.pow(dst, 2)
        ma = dst.max(dim=2, keepdim=True)[0]
        me = dst.mean(dim=2, keepdim=True)
        w = (ma - me) * (ma - me)
        dst = w * dst

        return dst.view(B,C, a,b)


    def apply_rarity(self, layer_output, layer_ind):
        features_processed = self.rarity_tensor(layer_output[layer_ind - 1])
        features_processed =features_processed.sum(dim=1)
        return F.interpolate(features_processed.clone().unsqueeze(0), (240,240), mode='bilinear', align_corners=False).squeeze(0)

    def fuse_itti_tensor(self, tensor):
        # Itti-like fusion between two maps
        B, C , W , H = tensor.shape
        tensor= tensor.view(B, C, -1)

        # Normalize 0 1
        tensor_min = tensor.min(dim=2, keepdim=True)[0]
        tensor_max = tensor.max(dim=2, keepdim=True)[0]
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)

        # Compute Weights
        tensor_max = tensor.max(dim=2, keepdim=True)[0]
        tensor_mean = tensor.mean(dim=2, keepdim=True)
        w1 = torch.square(tensor_max - tensor_mean)
        tensor = w1 * tensor
        tensor= tensor.sum(dim=1)

        # normalize 0 255
        tensor_min = tensor.min(dim=1, keepdim=True)[0]
        tensor_max = tensor.max(dim=1, keepdim=True)[0]
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)
        tensor *= 255

        return tensor.view(B,W,H)
    
    def forward(self, layer_output):
        layer1 = self.apply_rarity(layer_output, 1)
        layer2 = self.apply_rarity(layer_output, 2)
        layer4 = self.apply_rarity(layer_output, 4)
        layer5 = self.apply_rarity(layer_output, 5)
        layer7 = self.apply_rarity(layer_output, 11)
        layer8 = self.apply_rarity(layer_output, 12)
        layer9 = self.apply_rarity(layer_output, 13)
        layer11 = self.apply_rarity(layer_output, 18)
        layer12 = self.apply_rarity(layer_output, 19)
        layer13 = self.apply_rarity(layer_output, 20)
        layer15 = self.apply_rarity(layer_output, 28)
        layer16 = self.apply_rarity(layer_output, 29)
        layer17 = self.apply_rarity(layer_output, 30)

        high_level = self.fuse_itti_tensor(torch.stack([layer16, layer17, layer15],dim=1))
        medium_level2 = self.fuse_itti_tensor(torch.stack([layer12, layer13, layer11], dim=1))
        medium_level1 = self.fuse_itti_tensor(torch.stack([layer8, layer9, layer7], dim=1))
        low_level2 = self.fuse_itti_tensor(torch.stack([layer4, layer5], dim=1))

        low_level1 = self.fuse_itti_tensor(torch.stack([layer1, layer2], dim=1))

        groups= torch.stack([low_level1,low_level2,medium_level1,medium_level2,high_level ], dim=1)

        SAL = groups.sum(dim= 1)
        return SAL, groups


class DeepRareTorch(nn.Module):
    def __init__(self, threshold= 0.6):
        super(DeepRareTorch, self).__init__()
        # Load the VGG16 model

        self.model = models.vgg16(pretrained=True)
        self.model.eval()
        self.threshold= threshold


        self._face = (
            0
        )  # by default large faces detector is used
        self._margin = (
            0
        )  # by default additional margins are not added for images (good for classical images)

        self.rarity_network= RarityNetwork(threshold= self.threshold)

    def _get_margin(self):
        """Read width"""
        return self._margin

    def _set_margin(self, new_margin):
        """Modify witdh"""
        self._margin = new_margin

    margin = property(_get_margin, _set_margin)

    def _get_face(self):
        """Read width"""
        return self._face

    def _set_face(self, new_face):
        """Modify witdh"""
        self._face = new_face

    face = property(_get_face, _set_face)

    def forward(
            self,
            img_tensor,
            target_size= None,
        ):

        B,C,W,H = img_tensor.shape
        if target_size is None:
            target_size= (W,H)

        # prepare margins
        if self.margin == 1:
            img_tensor = F.interpolate(img_tensor, size=(168, 168), mode='bilinear', align_corners=False)
            img_tensor = F.pad(img_tensor, (28, 28, 28, 28), mode='circular')

        if self.margin == 0:
            img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bilinear', align_corners=False)

        # Get the outputs from all layers
        layer_names = [str(i) for i in range(1, 31)]  # Layers 1 to 17
        intermediate_layer_getter = IntermediateLayerGetter(self.model, layer_names)
        with torch.no_grad():
            layer_outputs = intermediate_layer_getter(img_tensor)

        SAL1, groups1 = self.rarity_network(layer_outputs)

        # # add face if needed
        # if self.face == 1:
        #     face_layer = self.get_faces(layer_outputs, 15)
        #     face_resize = cv2.resize(face_layer, (240, 240))



        #     if self.margin == 1:
        #         SAL = cv2.resize( SAL1 + face_resize, (224, 224))
        #         SAL = SAL[30:195, 30:195]

        #     if self.margin == 0:
        #         SAL = SAL1 + face_resize


        # if self.face == 0:
        if self.margin == 1:
            SAL = F.interpolate(SAL1, size=(224, 224), mode='bilinear', align_corners=False)
            SAL = SAL[30:195, 30:195]

        if self.margin == 0:
            SAL = SAL1

        SAL = F.interpolate(SAL.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)

        B,W,H= SAL.shape
        SAL= SAL.view(B,-1)
        tensor_min = SAL.min(dim=1, keepdim=True)[0]
        tensor_max = SAL.max(dim=1, keepdim=True)[0]
        SAL = (SAL - tensor_min) / (tensor_max - tensor_min + 1e-8)
        SAL= SAL.view(B,-1,W,H)


        groups1 = F.interpolate(groups1, size=target_size, mode='bilinear', align_corners=False)

        # face_resize = cv2.resize(face_resize, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        return SAL, groups1 #return saliency, groups and face
