
import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
import torchvision.transforms as transforms

import sys 
sys.path.append('..')
from model import DeepRare, UNIRARE, UNISAL
from data import FileOpener, O3Dataset, P3Dataset
from utils import metrics

def normalize_tensor(tensor, rescale=False):
    tmin = torch.min(tensor)
    if rescale or tmin < 0:
        tensor -= tmin
    tsum = tensor.sum()
    if tsum > 0:
        return tensor / tsum
    tensor.fill_(1. / tensor.numel())
    return tensor


def load_model(model_name, weights_path):
    if model_name == "unirare":
        model = UNIRARE(bypass_rnn=False)
    elif model_name == "unirare_finetuned":
        model = UNIRARE(bypass_rnn=False)
        if os.path.exists(weights_path + "weights_best.pth"):
            model.load_weights(weights_path + "weights_best.pth")
    elif model_name == "unisal":
        model = UNISAL(bypass_rnn=False)
    elif model_name == "deep_rare":
        model = DeepRare(
            # threshold=args.threshold,
            # model_name=args.model_name,
            # pretrained=True,
            # layers=args.layers_to_extract
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model


def post_process(map, img):
    print(img.shape)
    smap = np.exp(map)
    smap = np.squeeze(smap)
    smap = smap
    map_ = (smap / np.amax(smap) * 255).astype(np.uint8)
    return cv2.resize(map_ , (img.shape[-2] , img.shape[-1]))


def run_model(model, tensor_image, model_name):

    if model_name == "unirare" or model_name == "unirare_finetuned":
        tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)
        map_, SAL, groups = model(tensor_image, source="SALICON")

        SAL = SAL.squeeze(0)
        SAL = SAL - SAL.min()
        SAL = SAL / SAL.max()
        SAL = SAL.detach().cpu()

        return SAL.numpy() * 255

    elif model_name == "deep_rare":
        tensor_image = tensor_image.unsqueeze(0)
        SAL, groups  = model(tensor_image)

        SAL = SAL - SAL.min()
        SAL = SAL / SAL.max()

        return SAL.numpy() * 255
    
    elif model_name == "unisal":
        tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)

        map_ = model(tensor_image, source="SALICON")
        map_ = map_.squeeze(0).squeeze(0).squeeze(0).detach().cpu().numpy()
        map_ = post_process(map_, img)

        return map_
    
def parse_list_of_ints(value: str):
    """
    Parse une chaîne de caractères en liste d'entiers.

    Args:
        value (str): Chaîne de caractères à parser, format attendu: "1,2,3"

    Returns:
        list[int]: Liste d'entiers.
    """
    try:
        return [int(x) for x in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' n'est pas une liste d'entiers valide. Format attendu: '1,2,3'")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters test unisal  video/image")
    parser.add_argument(
        "--type", 
        type=str, 
        default="image", 
        help="image or video"
    )

    parser.add_argument(
        "--model", 
        type=str, 
        default="deep_rare", 
        choices=["unirare", "unirare_finetuned", "unisal", "deep_rare"],
        help="Select model to use"
    )

    parser.add_argument(
        "--P3Dataset", 
        type=str, 
        default="/Users/coconut/Documents/Dataset/Saliency/P3_data/", 
        help="path model to load"
    )

    parser.add_argument(
        "--O3Dataset", 
        type=str, 
        default="/Users/coconut/Documents/Dataset/Saliency/O3_data/", 
        help="path model to load"
    )

    parser.add_argument(
        "--MITDataset", 
        type=str, 
        default=".", 
        help="path model to load"
    )

    parser.add_argument(
        "--layers_to_extract", 
        type=parse_list_of_ints, 
        default="1,2,  4,5,8,  9,11,12,13,  16,17,18,19,  26,27,28,29", 
        help="Liste d'entiers séparés par des virgules (ex: 1,2,3)."
    )

    parser.add_argument(
        "--threshold", 
        type=float, 
        default=None, 
        help="Threshold for torch rare 2021"
    )

    args = parser.parse_args()

    # data results
    results = {}

    # Laod model
    model = load_model(args.model,"../model/weights/")

    # load dataloaders
    o3_dataset = O3Dataset(
        path =args.O3Dataset,
    )

    p3_dataset_sizes = P3Dataset(
        path =args.P3Dataset + "sizes/",
    )

    p3_dataset_orientations = P3Dataset(
        path =args.P3Dataset + "orientations/",
    )

    p3_dataset_colors = P3Dataset(
        path =args.P3Dataset + "colors/",
    )

    print(len(o3_dataset))

    # parcours de chaque dataset O3
    for i in range(0 , len(p3_dataset_colors)):
        img, targ, dist, _ = p3_dataset_colors[i]

        # run model 
        sal = run_model(model, img, args.model)
        print(sal.shape)

        # resize sal
        sal = cv2.resize(sal, (img.shape[-2], img.shape[-1]))

        # show shape
        print(sal.shape)
        print(targ.shape)
        print(dist.shape)

        # show image
        plt.figure()
        plt.subplot(141)
        plt.imshow(np.transpose(img, (1, 2, 0)))

        plt.subplot(142)
        plt.imshow(sal)

        plt.subplot(143)
        plt.imshow(np.transpose(targ, (1, 2, 0)))

        plt.subplot(144)
        plt.imshow(np.transpose(dist, (1, 2, 0)))

        plt.show()

        # show max and min
        print("sal : " , np.amax(sal) )
        print("targ : " , np.amax(targ.numpy()) )
        print("dist : " , np.amax(dist.numpy()) )

        print("sal : " , np.amin(sal) )
        print("targ : " , np.amin(targ.numpy()) )
        print("dist : " , np.amin(dist.numpy()) )

        # compute metrics 
        msr = metrics.compute_msr(sal, targ.squeeze(0).numpy(), dist.squeeze(0).numpy())
        print(msr)


        break

