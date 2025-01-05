import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

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
            layer_output = self.feature_extractor(input_image)

            return self.rarity(layer_output)


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
        resize_transform = T.Resize(size, interpolation=T.InterpolationMode.BILINEAR)
        return resize_transform(tensor.unsqueeze(0)).squeeze(0)

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

    def rarity(self, channel, bins=6):
        """
        Compute the single-resolution rarity for a given channel.

        Args:
            channel (torch.Tensor): Input channel.
            bins (int): Number of bins for histogram computation.

        Returns:
            torch.Tensor: Rarity map.
        """
        a, b = channel.shape

        # Apply border padding
        channel[:1, :] = 0
        channel[:, :1] = 0
        channel[a - 1:, :] = 0
        channel[:, b - 1:] = 0

        # Histogram computation
        channel = self.normalize_tensor(channel, min_val=0, max_val=256)
        hist = torch.histc(channel, bins=bins, min=0, max=256)
        hist = hist / hist.sum()
        hist = -torch.log(hist + 1e-4)

        # Back-projection
        hist_idx = ((channel * bins - 1).long().clamp(0, bins - 1))
        dst = self.normalize_tensor(hist[hist_idx], min_val=0, max_val=1)
        return self.map_ponderation(dst)

    def apply_rarity(self, layer_output):
        """
        Apply rarity computation to all feature maps in a layer.

        Args:
            layer_output (torch.Tensor): Feature maps of shape [B, C, H, W].
            threshold (float): Threshold to filter low-rarity values.

        Returns:
            torch.Tensor: Processed feature map.
        """
        feature_maps = layer_output.permute(0, 2, 3, 1)
        _, _, _, num_maps = feature_maps.shape

        processed_map = self.map_ponderation(self.rarity(feature_maps[0, :, :, 0]))

        for i in range(1, num_maps):
            feature = self.rarity(feature_maps[0, :, :, i])
            feature[:1, :] = 0
            feature[:, :1] = 0
            feature[-1:, :] = 0
            feature[:, -1:] = 0
            processed_map += self.map_ponderation(feature)

        processed_map = self.normalize_tensor(processed_map, min_val=0, max_val=1)
        if self.threshold is not None:
            processed_map[processed_map < self.threshold] = 0
        return processed_map

    def forward(self, layer_output):
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

        groups = torch.zeros((240, 240, len(packs)), device=layer_output[0].device)

        for i, pack in enumerate(packs):
            processed_layers = [
                self.tensor_resize(self.apply_rarity(features))
                for features in pack
            ]
            groups[:, :, i] = self.normalize_tensor(self.fuse_itti(processed_layers), min_val=0, max_val=256)

        return groups.sum(dim=-1), groups
