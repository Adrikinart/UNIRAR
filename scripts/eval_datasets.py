
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
import json
import time


if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")

print("DEFAULT_DEVICE " ,DEFAULT_DEVICE)

def load_model(args, weights_path):
    if args.model == "unirare":
        model = UNIRARE(bypass_rnn=False)
    elif args.model == "unirare_finetuned":
        model = UNIRARE(bypass_rnn=False)
        if os.path.exists(weights_path + "weights_best.pth"):
            model.load_weights(weights_path + "weights_best.pth")
    elif args.model == "unisal":
        model = UNISAL(bypass_rnn=False)
    elif args.model == "deep_rare":
        print("DEEP RARE")
        model = DeepRare(
            threshold=args.threshold,
            # model_name=args.model_name,
            pretrained=args.pretrained,
            layers=args.layers_to_extract
        )
    else:
        raise ValueError(f"Unknown model name: {args.model}")
    
    model= model.to(DEFAULT_DEVICE)
    return model

def post_process(map, img):
    smap = np.exp(map)
    smap = np.squeeze(smap)
    smap = smap
    map_ = (smap / np.amax(smap) * 255).astype(np.uint8)
    return cv2.resize(map_ , (img.shape[-2] , img.shape[-1]))

def run_model(model, image, model_name):
    tensor_image = image.to(DEFAULT_DEVICE)

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
        SAL = SAL.detach().cpu()

        return SAL.numpy() * 255
    
    elif model_name == "unisal":
        tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)

        map_ = model(tensor_image, source="SALICON")
        map_ = map_.squeeze(0).squeeze(0).squeeze(0).detach().cpu().numpy()
        map_ = post_process(map_, tensor_image)

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
    

def run_dataset(dataset, model, model_name):
    results = []
    total = len(dataset)-1
    start_time = time.time()
    
    for i in range(total):
        img, targ, dist, _ = dataset[i]

        # run model 
        start_time_model = time.time()
        sal = run_model(model, img, model_name)

        # resize sal
        sal = cv2.resize(sal, (targ.shape[-2], targ.shape[-1]))

        # # compute metrics 
        msr = metrics.compute_msr(sal, targ.squeeze(0).numpy(), dist.squeeze(0).numpy())

        # # Add time process for each image
        msr['process_time'] = time.time() - start_time_model
        results.append(msr)

        process_time = time.time() - start_time
        # Print loading bar with FPS information
        progress = (i + 1) / total
        bar_length = 40
        block = int(round(bar_length * progress))
        fps = (i + 1) / process_time
        text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}% | FPS: {fps:.2f} | {i}/{total}"
        sys.stdout.write(text)
        sys.stdout.flush()

        # break

    print()  # New line after progress bar is complete
    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters test unisal  video/image")
    parser.add_argument(
        "--type", 
        type=str, 
        default="image", 
        help="image or video"
    )

    parser.add_argument(
        "--input_size", 
        type=int, 
        default=412, 
    )

    parser.add_argument(
        "--model", 
        type=str, 
        default="unirare", 
        choices=["unirare", "unirare_finetuned", "unisal", "deep_rare"],
        help="Select model to use"
    )

    parser.add_argument(
        "--P3Dataset", 
        type=str, 
        default="C:/Users/538329/Desktop/PHD/RARE_UNISAL/DeepRare/Data/raw/P3_data/", 
        # default="C:/Users/lelon/Documents/Dataset/P3_data/", 

        help="path model to load"
    )

    parser.add_argument(
        "--O3Dataset", 
        type=str, 
        default="C:/Users/538329/Desktop/PHD/RARE_UNISAL/DeepRare/Data/raw/O3_data/", 
        # default="C:/Users/lelon/Documents/Dataset/O3_data/", 

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
        # default="6,7, 12,13,14,  16,17,18", 
        help="Liste d'entiers séparés par des virgules (ex: 1,2,3)."
    )

    parser.add_argument(
        "--threshold", 
        type=float, 
        default=None, 
        help="Threshold for torch rare 2021"
    )

    parser.add_argument(
        "--pretrained", 
        type=bool, 
        default=True, 
        help="True or False pretrained model"
    )

    args = parser.parse_args()

    # data results
    results = {}

    # Laod model
    model = load_model(args,"../model/weights/")

    # load dataloaders
    o3_dataset = O3Dataset(
        path =args.O3Dataset,
        input_size=(args.input_size,args.input_size)

    )

    p3_dataset_sizes = P3Dataset(
        path =args.P3Dataset + "sizes/",
        input_size=(args.input_size,args.input_size)
    )

    p3_dataset_orientations = P3Dataset(
        path =args.P3Dataset + "orientations/",
        input_size=(args.input_size,args.input_size)

    )

    p3_dataset_colors = P3Dataset(
        path =args.P3Dataset + "colors/",
        input_size=(args.input_size,args.input_size)

    )

    # Run dataset and collect results
    results['O3Dataset'] = run_dataset(o3_dataset, model, args.model)
    results['P3Dataset_sizes'] = run_dataset(p3_dataset_sizes, model, args.model)
    results['P3Dataset_orientations'] = run_dataset(p3_dataset_orientations, model, args.model)
    results['P3Dataset_colors'] = run_dataset(p3_dataset_colors, model, args.model)

    # Save results to JSON file
    results['args'] = vars(args)

    # Create a name based on args information
    result_filename = f"results_{args.model}_{args.type}.json"
    with open("../res/" + result_filename, 'w') as f:
        json.dump(results, f, indent=4)

