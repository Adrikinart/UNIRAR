import torchvision.models as models
import torch
import torch.nn as nn
from collections import OrderedDict
import sys
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from  scipy import ndimage

print(sys.path)
sys.path.append('../model/PNAS/')
print(sys.path)
from PNASnet import *
from genotypes import PNASNet
import torch.nn.functional as nnf
import numpy as np
from time import time
import torch.nn.functional as F

   
class PNASModel(nn.Module):

    def __init__(self, num_channels=3, train_enc=False, load_weight=1):
        super(PNASModel, self).__init__()
        self.pnas = NetworkImageNet(216, 1001, 12, False, PNASNet)
        if load_weight:
            self.pnas.load_state_dict(torch.load(self.path))

        for param in self.pnas.parameters():
            param.requires_grad = train_enc

        self.padding = nn.ConstantPad2d((0,1,0,1),0)
        self.drop_path_prob = 0

        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.deconv_layer0 = nn.Sequential(
            nn.Conv2d(in_channels = 4320, out_channels = 512, kernel_size=3, padding=1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )

        self.deconv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 512+2160, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 1080+256, out_channels = 270, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 540, out_channels = 96, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            self.linear_upsampling
        )
        self.deconv_layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )

    def forward(self, images):
        batch_size = images.size(0)

        s0 = self.pnas.conv0(images)
        s0 = self.pnas.conv0_bn(s0)
        out1 = self.padding(s0)

        s1 = self.pnas.stem1(s0, s0, self.drop_path_prob)
        out2 = s1
        s0, s1 = s1, self.pnas.stem2(s0, s1, 0)

        for i, cell in enumerate(self.pnas.cells):
            s0, s1 = s1, cell(s0, s1, 0)
            if i==3:
                out3 = s1
            if i==7:
                out4 = s1
            if i==11:
                out5 = s1

        out5 = self.deconv_layer0(out5)

        x = torch.cat((out5,out4), 1)
        x = self.deconv_layer1(x)

        x = torch.cat((x,out3), 1)
        x = self.deconv_layer2(x)

        x = torch.cat((x,out2), 1)
        x = self.deconv_layer3(x)
        x = torch.cat((x,out1), 1)

        x = self.deconv_layer4(x)
        
        x = self.deconv_layer5(x)
        x = x.squeeze(1)
     #   print("PNAS pred actual pnas:", x.mean(),x.min(), x.max(), x.sum())

        return x

class PNASVolModellast(nn.Module):

    def __init__(self, time_slices, num_channels=3, train_enc=False, load_weight=1):
            super(PNASVolModellast, self).__init__()

            self.pnas = NetworkImageNet(216, 1001, 12, False, PNASNet)
            if load_weight:            
                state_dict = torch.load(self.path)
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'module'  in k:
                        k = 'module.pnas.' + k
                    else:
                        k = k.replace('pnas.', '')
                    new_state_dict[k] = v
                self.pnas.load_state_dict(new_state_dict, strict=False)
               

            for param in self.pnas.parameters():
                param.requires_grad = train_enc

            self.padding = nn.ConstantPad2d((0,1,0,1),0)
            self.drop_path_prob = 0

            self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

            self.deconv_layer0 = nn.Sequential(
                nn.Conv2d(in_channels = 4320, out_channels = 512, kernel_size=3, padding=1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )

            self.deconv_layer1 = nn.Sequential(
                nn.Conv2d(in_channels = 512+2160, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
            self.deconv_layer2 = nn.Sequential(
                nn.Conv2d(in_channels = 1080+256, out_channels = 270, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
            self.deconv_layer3 = nn.Sequential(
                nn.Conv2d(in_channels = 540, out_channels = 96, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
            self.deconv_layer4 = nn.Sequential(
                nn.Conv2d(in_channels = 192, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )

            self.deconv_layer5 = nn.Sequential(
                nn.Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels = 32, out_channels = time_slices, kernel_size = 3, padding = 1, bias = True),
                nn.Sigmoid()
            )
        
    def forward(self, images):
        s0 = self.pnas.conv0(images)
        s0 = self.pnas.conv0_bn(s0)
        out1 = self.padding(s0)

        s1 = self.pnas.stem1(s0, s0, self.drop_path_prob)
        out2 = s1
        s0, s1 = s1, self.pnas.stem2(s0, s1, 0)

        for i, cell in enumerate(self.pnas.cells):
            s0, s1 = s1, cell(s0, s1, 0)
            if i==3:
                out3 = s1
            if i==7:
                out4 = s1
            if i==11:
                out5 = s1

        out5 = self.deconv_layer0(out5)

        x = torch.cat((out5,out4), 1)
        x = self.deconv_layer1(x)

        x = torch.cat((x,out3), 1)
        x = self.deconv_layer2(x)

        x = torch.cat((x,out2), 1)
        x = self.deconv_layer3(x)
        x = torch.cat((x,out1), 1)

        x = self.deconv_layer4(x)

        x = self.deconv_layer5(x)
        x = x / x.max()

        return x , [out1,out2,out3,out4,out5]   
    
class PNASBoostedModelMultiLevel(nn.Module):

    def __init__(self, device, model_path, model_vol_path, time_slices, train_model=False, selected_slices=""):
        super(PNASBoostedModelMultiLevel, self).__init__()

        
        self.selected_slices = selected_slices

            
        self.linear_upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

        self.deconv_layer1 = nn.Sequential(
                nn.Conv2d(in_channels = 512+2160+6, out_channels = 256, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
        self.deconv_layer2 = nn.Sequential(
                nn.Conv2d(in_channels = 1080+256+6, out_channels = 270, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
        self.deconv_layer3 = nn.Sequential(
                nn.Conv2d(in_channels = 540+6, out_channels = 96, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
        self.deconv_layer4 = nn.Sequential(
                nn.Conv2d(in_channels = 192+6, out_channels = 128, kernel_size = 3, padding = 1, bias = True),
                nn.ReLU(inplace=True),
                self.linear_upsampling
            )
        

        self.deconv_mix = nn.Sequential(
            nn.Conv2d(in_channels = 128+6 , out_channels = 16, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = 3, padding = 1, bias = True),
            nn.Sigmoid()
        )
        model_vol = PNASVolModellast(time_slices=5, load_weight=0) #change this to time slices
        model_vol = nn.DataParallel(model_vol).cuda()
        model_path = model_path.replace('./', '../model/') # update path
        state_dict = torch.load(model_path)
        vol_state_dict = OrderedDict()
        sal_state_dict = OrderedDict()
        smm_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            if 'pnas_vol'  in k:

                k = k.replace('pnas_vol.module.', '')
                vol_state_dict[k] = v
            elif 'pnas_sal'  in k:
                k = k.replace('pnas_sal.module.', '')
                sal_state_dict[k] = v
            else:
                smm_state_dict[k] = v
                
        self.load_state_dict(smm_state_dict)
        model_vol.load_state_dict(vol_state_dict)
        self.pnas_vol = nn.DataParallel(model_vol).cuda()

        for param in self.pnas_vol.parameters():
            param.requires_grad = False


        model = PNASModel(load_weight=0)
        model = nn.DataParallel(model).cuda()

        model.load_state_dict(sal_state_dict, strict=True)
        self.pnas_sal = nn.DataParallel(model).to(device)

        for param in self.pnas_sal.parameters():
            param.requires_grad = False #train_model
        

        self.extracted_layers = nn.ModuleList()

    def initialize_forward_layers(self, layer_indexes):
        """
        Initialize the `layers` feature with specified convolutional layers from the encoder.
        Args:
            layer_indexes (list of int): List of indexes of the convolutional layers to extract.
        """
        self.layers = nn.ModuleList()  # Reset layers
        all_layers = list(self.pnas_sal.module.pnas.children())  # Access layers from PNAS encoder
        print(f"first 5 layers: '\n' {all_layers[:5]}")

        # Append selected layers based on input indexes
        for idx in layer_indexes:
            if idx < len(all_layers):
                self.layers.append(all_layers[idx])
            else:
                raise IndexError(f"Layer index {idx} is out of bounds. Total layers: {len(all_layers)}")

    def forward(self, images):
        print("IMAGES", images.shape) # IMAGES torch.Size([1, 3, 412, 412])

        pnas_pred = self.pnas_sal(images).unsqueeze(1) 
        pnas_vol_pred , outs = self.pnas_vol(images)
        print(pnas_vol_pred.shape)

        out1 , out2, out3, out4, out5 = outs

        # Forward pass through selected layers and extract activations
        layer_outputs = [layer(images) for layer in self.layers]

        x_maps = torch.cat((pnas_pred, pnas_vol_pred), 1)

        x = torch.cat((out5,out4), 1)
        x_maps16 = nnf.interpolate(x_maps, size=(16, 16), mode='bicubic', align_corners=False)

        x = torch.cat((x,x_maps16), 1)

        x = self.deconv_layer1(x)
        x = torch.cat((x,out3), 1)
        x_maps32 = nnf.interpolate(x_maps, size=(32, 32), mode='bicubic', align_corners=False)
        x = torch.cat((x,x_maps32), 1)

        x = self.deconv_layer2(x)
        x = torch.cat((x,out2), 1)
        x_maps64 = nnf.interpolate(x_maps, size=(64, 64), mode='bicubic', align_corners=False)
        x = torch.cat((x,x_maps64), 1)

        x = self.deconv_layer3(x)
        x = torch.cat((x,out1), 1)
        x_maps128 = nnf.interpolate(x_maps, size=(128, 128), mode='bicubic', align_corners=False)

        x = torch.cat((x,x_maps128), 1)

        x = self.deconv_layer4(x)
        x = torch.cat((x,x_maps), 1)
        
        x = self.deconv_mix(x)
      
        x = x.squeeze(1)

        return x, pnas_vol_pred, layer_outputs
    
class RarityNetwork(nn.Module):
    def __init__(self, threshold= None):
        super(RarityNetwork, self).__init__()
        self.threshold= threshold

    def rarity_tensor(self, channel):
        B,C , a , b = channel.shape
        if a > 10:  # manage margins for high-level features
            channel[:,:,0:1, :] = 0
            channel[:,:,:, a - 1:a] = 0
            channel[:,:,:, 0:1] = 0
            channel[:,:,b - 1:b, :] = 0
        elif a > 28:  # manage margins for mid-level features
            channel[:,:,0:2, :] = 0
            channel[:,:,:, a - 2:a] = 0
            channel[:,:,:, 0:2] = 0
            channel[:,:,b - 2:b, :] = 0
        elif a > 50:  # manage margins for low-level features
            channel[:,:,0:3, :] = 0
            channel[:,:,:, a - 3:a] = 0
            channel[:,:,:, 0:3] = 0
            channel[:,:,b - 3:b, :] = 0
        
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

        if a > 28:
            dst[:,:,0:2, :] = 0
            dst[:,:,:, a - 2:a] = 0
            dst[:,:,:, 0:2] = 0
            dst[:,:,b - 2:b, :] = 0
        elif a > 50:
            dst[:,:,0:3, :] = 0
            dst[:,:,:, a - 3:a] = 0
            dst[:,:,:, 0:3] = 0
            dst[:,:,b - 3:b, :] = 0

        dst = dst.view(B,C, -1)
        dst = torch.pow(dst, 2)
        ma = dst.max(dim=2, keepdim=True)[0]
        me = dst.mean(dim=2, keepdim=True)
        w = (ma - me) * (ma - me)
        dst = w * dst

        return dst.view(B,C, a,b)


    def apply_rarity(self, layer_output, layer_ind):
        features_processed = self.rarity_tensor(layer_output[layer_ind - 1].clone())

        features_processed =features_processed.sum(dim=1)

        features_processed= F.interpolate(features_processed.clone().unsqueeze(0), (240,240), mode='bilinear', align_corners=False)

        return features_processed

    def fuse_itti_tensor(self, tensor):
        # Itti-like fusion between two maps
        B, C , W , H = tensor.shape
        tensor= tensor.view(B, C, -1)

        # Normalize 0 1
        tensor_min = tensor.min(dim=2, keepdim=True)[0]
        tensor_max = tensor.max(dim=2, keepdim=True)[0]
        tensor = (tensor - tensor_min) / (tensor_max - tensor_min + 1e-8)

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
    
    def forward(
            self,
            layer_output,
            layers=[
                [4,5],
                [7,8],
                [10,11],
                [13,14],
                [16,17]
            ]
        ):

        # for i,layer in enumerate(layer_output):
        #     print(f"Layer {i+1}: ",layer.shape)

        groups = []
        for layers_index in layers:
            tempo = []
            for index in layers_index:
                tempo.append(self.apply_rarity(layer_output, index))

            tempo = torch.cat(tempo, dim=1)
            # tempo = torch.stack(tempo, dim=1)
            groups.append(self.fuse_itti_tensor(tempo))
        groups= torch.stack(groups, dim=1)

        
        SAL = groups.sum(dim= 1)

        B,W,H = SAL.shape
        SAL = SAL.view(B , -1)

        SAL_min = SAL.min(dim=1, keepdim=True)[0]
        SAL_max = SAL.max(dim=1, keepdim=True)[0]
        SAL = (SAL - SAL_min) / (SAL_max - SAL_min + 1e-8)

        SAL = torch.exp(SAL)

        SAL_min = SAL.min(dim=1, keepdim=True)[0]
        SAL_max = SAL.max(dim=1, keepdim=True)[0]
        SAL = (SAL - SAL_min) / (SAL_max - SAL_min + 1e-8)


        SAL= SAL.view(B,W,H)

        return SAL, groups


class BaseModel(nn.Module):
    """Abstract model class with functionality to save and load weights"""

    def forward(self, *input):
        raise NotImplementedError

    def save_weights(self, directory, name):
        torch.save(self.state_dict(), directory / f"weights_{name}.pth")

    def load_weights(self, file , DEFAULT_DEVICE = torch.device("cpu")):
        self.load_state_dict(
            torch.load(file, map_location=DEFAULT_DEVICE)
        )

    def load_epoch_checkpoint(self, directory, epoch):
        """Load state_dict from a Trainer checkpoint at a specific epoch"""
        chkpnt = torch.load(directory / f"chkpnt_epoch{epoch:04d}.pth" , map_location = 'cpu' ,  weights_only=True)
        self.load_state_dict(chkpnt["model_state_dict"] , map_location = 'cpu' ,  weights_only=True)


class TempSalRare(BaseModel):
    def __init__(self, device, model_path, model_vol_path, time_slices, train_model=False, selected_slices=""):
        super(TempSalRare, self).__init__()
        self.pnas = PNASBoostedModelMultiLevel(device, model_path, model_vol_path, time_slices, train_model, selected_slices)
        self.deeprare = RarityNetwork(threshold=0.5)



    def forward(self, x , target_size = None):
        if target_size is None:
            target_size = x.shape[-2:]
        elif type(target_size) is not tuple:
            if target_size[0].shape != 1:
                target_size = [target_size[0][0] , target_size[1][0]]
        
        sal_rare = []
        level_rare = []
        pnas_preds = []
        pnas_vol_preds = []
        for t, img in enumerate(torch.unbind(x, dim=1)):
            pnas_pred, pnas_vol_pred, layer_outputs = self.pnas(img)
            sal, groups = self.deeprare(layer_outputs)
            sal_rare.append(sal)
            level_rare.append(groups)
            pnas_preds.append(pnas_pred)
            pnas_vol_preds.append(pnas_vol_pred)
        
        sal_rare = torch.stack(sal_rare, dim=1)
        level_rare = torch.stack(level_rare, dim=1)
        pnas_preds = torch.stack(pnas_preds, dim=1)
        pnas_vol_preds = torch.stack(pnas_vol_preds, dim=1)


        return sal_rare, level_rare, pnas_preds, pnas_vol_preds





        
