import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt

class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_indices):
        """
        Initialize the feature extractor.

        Args:
            model: The pre-trained model (e.g., VGG16).
            layer_indices: List of indices of layers to extract features from.
        """
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.layer_indices = layer_indices
    
    def forward(self, x):
        features = []
        for idx, layer in enumerate(self.model.features):
            x = layer(x)
            if idx in self.layer_indices:
                features.append(x)
        return features

import time
class DeepRare(nn.Module):
    def __init__(
            self,
            threshold= None,
            model_name='mobilenet_v2',
            pretrained=True,
            layers=[1,2,  4,5,8,  9,11,12,13,  16,17,18,19,  26,27,28,29]
        ):
        """
        Constructor for the DeepRare model.
        """
        super(DeepRare, self).__init__()
        self.threshold = threshold

        print(f"Chargement du modèle pré-entraîné : {model_name}")
        if model_name.lower() == "vgg16":
            self.model = models.vgg16(pretrained=pretrained)
        elif model_name.lower() == "mobilenet_v2":
            self.model = models.mobilenet_v2(pretrained=pretrained)
        else:
            raise ValueError(f"Modèle non supporté : {model_name}")
        
        self.model.eval()
        self.feature_extractor = FeatureExtractor(self.model, layers)

        self.rarity = RarityNetwork()
        
    def forward(self, input_image):
            """
            Forward pass to process feature maps.

            Args:
                input_image (torch.Tensor): Input image tensor.

            Returns:
                torch.Tensor: Fused saliency map.
                torch.Tensor: Stacked feature maps.
            """
            target_size = input_image.shape[-2:]

            layer_output = self.feature_extractor(input_image)
            return self.rarity(layer_output, target_size)
class RarityNetwork(nn.Module):
    """
    DeepRare2019 Class.
    """

    def __init__(
            self,
            threshold= None,
            layers=[1,2,  4,5,8,  9,11,12,13,  16,17,18,19,  26,27,28,29]
        ):
        """
        Constructor for the DeepRare model.
        """
        super(RarityNetwork, self).__init__()
        self.threshold = threshold

    @staticmethod
    def tensor_resize(tensor, size=(240, 240)):
        """
        Resize a tensor to the specified size using bilinear interpolation.

        Args:
            tensor (torch.Tensor): Input tensor.
            size (tuple): Desired output size (height, width).

        Returns:
            torch.Tensor: Resized tensor.
        """

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch dimension if missing
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)  # Add batch dimension if missing
        elif tensor.dim() != 4:
            raise ValueError("Input tensor must have 3 or 4 dimensions")
        
        tensor=  F.interpolate(tensor, size=size, mode="bilinear", align_corners=False).squeeze(0)

        return tensor

    @staticmethod
    def normalize_tensor(tensor, min_val=0, max_val=1):
        """
        Normalize a tensor to the specified range [min_val, max_val].

        Args:
            tensor (torch.Tensor): Input tensor.
            min_val (float): Minimum value of the normalized range.
            max_val (float): Maximum value of the normalized range.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        tensor_min = tensor.min()
        tensor_max = tensor.max()
        if tensor_max - tensor_min == 0:
            return torch.zeros_like(tensor)
        return ((tensor - tensor_min) / (tensor_max - tensor_min)) * (max_val - min_val) + min_val
    
    
    def map_ponderation(self, tensor):
        """
        Apply weighting to a tensor map based on its rarity.

        Args:
            tensor (torch.Tensor): Input tensor map.

        Returns:
            torch.Tensor: Weighted tensor map.
        """
        map_max = tensor.max()
        map_mean = tensor.mean()
        map_weight = (map_max - map_mean) ** 2
        return self.normalize_tensor(tensor, min_val=0, max_val=1) * map_weight
    
    def map_ponderation_tensor(self,tensor):
        """
        Apply weighting to a tensor map based on its rarity for each channel independently.

        Args:
            tensor (torch.Tensor): Input tensor of shape [C, W].

        Returns:
            torch.Tensor: Weighted tensor map of shape [C, W].
        """
        map_max = tensor.max(dim=1, keepdim=True)[0]
        map_mean = tensor.mean(dim=1, keepdim=True)
        map_weight = (map_max - map_mean) ** 2

        tensor_min = tensor.min(dim=1, keepdim=True)[0]
        tensor_max = tensor.max(dim=1, keepdim=True)[0]
        normalized_tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)

        return normalized_tensor * map_weight

    def fuse_itti(self, maps):
        """
        Perform Itti-like fusion of maps.

        Args:
            maps (list[torch.Tensor]): List of input maps to fuse.

        Returns:
            torch.Tensor: Fused map.
        """

        fused_map = torch.zeros_like(maps[0])
        for feature_map in maps:
            fused_map += self.map_ponderation(feature_map)
        return fused_map


    def rarity(self, feature_maps, bins=6):
        """
        Compute the single-resolution rarity for a given data.

        Args:
            data (torch.Tensor): Input data of shape [B, C, W, H].
            bins (int): Number of bins for histogram computation.

        Returns:
            torch.Tensor: Rarity map.
        """

        B, C, W, H = feature_maps.shape

        batchs= []

        for bidx in range(B):

            feature_maps_histo_batch = feature_maps[bidx, :, :, :]

            feature_maps_histo_batch[:, :1, :] = 0
            feature_maps_histo_batch[:, :, :1] = 0
            feature_maps_histo_batch[:, W - 1:, :] = 0
            feature_maps_histo_batch[:, :, H - 1:] = 0
            feature_maps_histo_batch = feature_maps_histo_batch.view(C, -1)


            tensor_min = feature_maps_histo_batch.min(dim=1, keepdim=True)[0]
            tensor_max = feature_maps_histo_batch.max(dim=1, keepdim=True)[0]
            feature_maps_histo_batch = (feature_maps_histo_batch - tensor_min) / (tensor_max - tensor_min + 1e-8)
            feature_maps_histo_batch = feature_maps_histo_batch * 256


            min_val, max_val = feature_maps_histo_batch.min(), feature_maps_histo_batch.max()
            bin_edges = torch.linspace(min_val, max_val, steps=bins + 1, device=feature_maps_histo_batch.device)

            bin_indices = torch.bucketize(feature_maps_histo_batch, bin_edges, right=True) - 1

            bin_indices = bin_indices.clamp(0, bins - 1)

            histograms = torch.zeros((feature_maps_histo_batch.size(0), bins), device=feature_maps_histo_batch.device)
            histograms.scatter_add_(1, bin_indices, torch.ones_like(bin_indices, dtype=torch.float, device=feature_maps_histo_batch.device))    

            histograms = histograms / histograms.sum(dim=1, keepdim=True)
            histograms = -torch.log(histograms + 1e-4)
            hists_idx = ((feature_maps_histo_batch * bins - 1).long().clamp(0, bins - 1))
            channels = histograms.gather(1, hists_idx)

            tensor_min = channels.min(dim=1, keepdim=True)[0]
            tensor_max = channels.max(dim=1, keepdim=True)[0]
            channels = (channels - tensor_min) / (tensor_max - tensor_min + 1e-8)

            channels = self.map_ponderation_tensor(channels)

            hists_idx = hists_idx.view(C, H, W)
            channels = channels.view(C, H, W)

            channels[:,:1, :] = 0
            channels[:,:, :1] = 0
            channels[:,-1:, :] = 0
            channels[:,:, -1:] = 0
        
            channels = channels.view(C,-1)
            channels = self.map_ponderation_tensor(channels)

            channels= channels.sum(dim=0)

            tensor_min = channels.min(dim=0, keepdim=True)[0]
            tensor_max = channels.max(dim=0, keepdim=True)[0]
            channels = (channels - tensor_min) / (tensor_max - tensor_min + 1e-8)
            
            batchs.append(channels.view(H, W))

        return torch.stack(batchs, dim=0)
    

    def apply_rarity(self, layer_output):
        """
        Apply rarity computation to all feature maps in a layer.

        Args:
            layer_output (torch.Tensor): Feature maps of shape [B, C, H, W].
            threshold (float): Threshold to filter low-rarity values.

        Returns:
            torch.Tensor: Processed feature map.
        """
        processed_map = self.rarity(layer_output.clone())

        if self.threshold is not None:
            processed_map[processed_map < self.threshold] = 0


        # add fuse itti 

        # normalize tensor

        # add resize

        return processed_map

    def forward(self, layer_output, target_size):
        """
        Forward pass to process feature maps.

        Args:
            layer_output (torch.Tensor): Input layers.

        Returns:
            torch.Tensor: Fused saliency map.
            torch.Tensor: Stacked feature maps.
        """

        packs = []
        for layer in layer_output:
            added = next((pack for pack in packs if pack[0].shape[-2:] == layer.shape[-2:]), None)
            if added:
                added.append(layer)
            else:
                packs.append([layer])

        fused_maps = [] 
        for pack in packs:

            

            processed_layers = [
            self.tensor_resize(self.apply_rarity(features), target_size)
            for features in pack
            ]

            fused_map = self.normalize_tensor(self.fuse_itti(processed_layers), min_val=0, max_val=256)

            fused_maps.append(fused_map)

        fused_maps = torch.stack(fused_maps, dim=-1)
        result = fused_maps.sum(dim=-1)

        return result, fused_maps
