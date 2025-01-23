
import os
import cv2
import numpy as np
import argparse
import sys 
import time

sys.path.append('..')
from utils import metrics

import torch
import matplotlib.pyplot as plt
from model import RarityNetwork

from utils.helper import show_saliency
from model import load_model, load_dataloader, run_model  

if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("cpu")
else:
    DEFAULT_DEVICE = torch.device("cpu")


if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser(description="Parameters test unisal  video/image")
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=None, 
        help="Threshold for torch rare 2021"
    )

    parser.add_argument(
        "--finetune", 
        type=str, 
        default="true",
        help="true for finetune, false for no finetune",
        choices = ["true", "false"]
    )



    parser.add_argument(
        "--directory", 
        type=str, 
        # default="/Users/coconut/Documents/Dataset/Saliency/SALICON/test/",
        # default="C:/Users/lelon/Documents/Dataset/Salicon/test/",
        default="./inputs/datasets/mit1003/worsts/",
        help="path directory images"
    )

    parser.add_argument(
        "--show", 
        type=str, 
        default="false",
        help="true for show, false for no ",
        choices = ["true", "false"]
    )

    models = ["Unisal", "TranSalNetDense" , "TranSalNetRes" , "TempSal"]



    # get arguments
    args = parser.parse_args()



    # rarity network
    rarity_model= RarityNetwork(
        threshold=args.threshold
    )
    

    files = os.listdir(args.directory)
    start_time_global = time.time()
    
    for index, filename in enumerate(files):
        go_path = os.path.join(args.directory , filename)

        # open images
        img = cv2.imread(go_path)
        if img is None:
            continue

        maps = {}

        for model_name in models:
            args.model = model_name

            # load model
            model, layers_index = load_model(args, DEFAULT_DEVICE)

            # load file opener
            file_opener = load_dataloader(args)

            # load model on default device
            rarity_model= rarity_model.to(DEFAULT_DEVICE)
            model = model.to(DEFAULT_DEVICE)

            # run model
            start_time = time.time()
            saliency, layers = run_model(args, model,file_opener ,go_path, DEFAULT_DEVICE)

            # run rarity network
            rarity_map, groups = rarity_model(
                layers_input= layers,
                layers_index=layers_index
            ) 

            process_time = time.time() - start_time


            # create and save maps
            maps[model_name] = {
                'saliency': saliency,
                'rarity': rarity_map,
            }

            maps[model_name]['saliency_Add']= rarity_model.add_rarity(saliency, rarity_map)
            maps[model_name]['saliency_Sub']= rarity_model.sub_rarity(saliency, rarity_map)
            maps[model_name]['saliency_Prod']= rarity_model.prod_rarity(saliency, rarity_map)
            maps[model_name]['saliency_Itti']= rarity_model.fuse_rarity(saliency, rarity_map)

        fig, axs = plt.subplots(len(models), 5, figsize=(20, 10))
        for i, model_name in enumerate(models):
            axs[i, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[i, 0].set_title(f'{model_name} Original Image')
            axs[i, 0].axis('off')
            alpha = 0.4  # Transparency factor

            saliency = maps[model_name]['saliency'].permute(1,2,0).detach().cpu().numpy()
            add = maps[model_name]['saliency_Add'].permute(1,2,0).detach().cpu().numpy()
            prod = maps[model_name]['saliency_Prod'].permute(1,2,0).detach().cpu().numpy()
            itti= maps[model_name]['saliency_Itti'].permute(1,2,0).detach().cpu().numpy()

            # resize all maps to the same size
            saliency = cv2.resize(saliency, (img.shape[1], img.shape[0]))
            add = cv2.resize(add, (img.shape[1], img.shape[0]))
            prod = cv2.resize(prod, (img.shape[1], img.shape[0]))
            itti = cv2.resize(itti, (img.shape[1], img.shape[0]))

            # cast
            saliency = cv2.normalize(saliency, None, 0, 255, cv2.NORM_MINMAX)
            add = cv2.normalize(add, None, 0, 255, cv2.NORM_MINMAX)
            prod = cv2.normalize(prod, None, 0, 255, cv2.NORM_MINMAX)
            itti = cv2.normalize(itti, None, 0, 255, cv2.NORM_MINMAX)

            saliency = saliency.astype(np.uint8)
            add = add.astype(np.uint8)
            prod = prod.astype(np.uint8)
            itti = itti.astype(np.uint8)

            # jetmap  all maps
            saliency = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
            add = cv2.applyColorMap(add, cv2.COLORMAP_JET)
            prod = cv2.applyColorMap(prod, cv2.COLORMAP_JET)
            itti = cv2.applyColorMap(itti, cv2.COLORMAP_JET)

            #  invert rgb channel for maps 
            saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2RGB)
            add = cv2.cvtColor(add, cv2.COLOR_BGR2RGB)
            prod = cv2.cvtColor(prod, cv2.COLOR_BGR2RGB)
            itti = cv2.cvtColor(itti, cv2.COLOR_BGR2RGB)


            # add weights on img for each maps 
            saliency = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha, saliency, 1 - alpha, 0)
            add = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha, add, 1 - alpha, 0)
            prod = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha, prod, 1 - alpha, 0)
            itti = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), alpha, itti, 1 - alpha, 0)

            #  subplot each maps
            axs[i, 1].imshow(saliency)
            axs[i, 1].set_title(f'{model_name} Saliency')
            axs[i, 1].axis('off')

            axs[i, 2].imshow(add)
            axs[i, 2].set_title(f'{model_name} Saliency + Rarity')
            axs[i, 2].axis('off')

            axs[i, 3].imshow(prod)
            axs[i, 3].set_title(f'{model_name} Saliency * Rarity')
            axs[i, 3].axis('off')

            axs[i, 4].imshow(itti)
            axs[i, 4].set_title(f'{model_name} Saliency Itti')
            axs[i, 4].axis('off')

        plt.tight_layout()
        plt.show()

        # break