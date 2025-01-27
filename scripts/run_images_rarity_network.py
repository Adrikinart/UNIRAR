
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
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")



def run_dataset(args, saliency_model, rarity_model, layers_index, file_opener):
    files = os.listdir(args.directory)
    start_time_global = time.time()
    
    for index, filename in enumerate(files):
        go_path = os.path.join(args.directory , filename)

        # open images
        img = cv2.imread(go_path)
        if img is None:
            continue

        # run model
        start_time = time.time()
        saliency, layers = run_model(args, saliency_model,file_opener ,go_path, DEFAULT_DEVICE)

        # run rarity network
        rarity_map, groups = rarity_model(
            layers_input= layers,
            layers_index=layers_index
        ) 

        process_time = time.time() - start_time


        # create and save maps
        maps = {
            'saliency': saliency,
            'rarity': rarity_map,
        }

        maps['saliency_Add']= rarity_model.add_rarity(saliency, rarity_map)
        maps['saliency_Sub']= rarity_model.sub_rarity(saliency, rarity_map)
        maps['saliency_Prod']= rarity_model.prod_rarity(saliency, rarity_map)
        maps['saliency_Itti']= rarity_model.fuse_rarity(saliency, rarity_map)

        
         # writes maps
        for key, v in maps.items():
            os.makedirs(args.path + "/" +key, exist_ok=True)

            v = v.squeeze(0).squeeze(0).detach().cpu().numpy()
            v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
            v = cv2.resize(v, (img.shape[1], img.shape[0]))
            v = v.astype(np.uint8)

            # save images in folder 
            file_name = filename.replace(".jpg" , ".png")
            cv2.imwrite(f"{args.path}/{key}/{file_name}", v)


        # Print loading bar with FPS information
        process_time_global = time.time() - start_time_global

        total = len(files)
        progress = (index + 1) / total
        bar_length = 40
        block = int(round(bar_length * progress))
        fps = (index + 1) / process_time_global
        text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}% | FPS: {fps:.2f} | {index}/{total} | Time: {process_time:.4f}s"
        sys.stdout.write(text)
        sys.stdout.flush()

        # # show results
        if args.show == "true":
            show_saliency(
                img= img,
                maps= maps,
                details= groups
            )
            plt.show()


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
        "--model", 
        type=str, 
        default="Unisal", 
        help="select model",
        choices= ["Unisal", "TranSalNetDense" , "TranSalNetRes" , "TempSal"]
    )

    parser.add_argument(
        "--path", 
        type=str, 
        # default="./outputs/inputs_images",
        default="./outputs/salicon_data_test",
        help="path to save results"
    )

    parser.add_argument(
        "--directory", 
        type=str, 
        # default="/Users/coconut/Documents/Dataset/Saliency/SALICON/test/",
        # default="C:/Users/lelon/Documents/Dataset/Salicon/test/",
        default="./inputs/images/",
        help="path directory images"
    )

    parser.add_argument(
        "--show", 
        type=str, 
        default="false",
        help="true for show, false for no ",
        choices = ["true", "false"]
    )


    # get arguments
    args = parser.parse_args()

    # rarity network
    rarity_model= RarityNetwork(
        threshold=args.threshold
    )

    # load model
    model, layers_index = load_model(args, DEFAULT_DEVICE)

    # load file opener
    file_opener = load_dataloader(args)

    # load model on default device
    rarity_model= rarity_model.to(DEFAULT_DEVICE)
    model = model.to(DEFAULT_DEVICE)
    args.path+= f"_{args.model}_finetune_{args.finetune.lower()}_threshold_{args.threshold}/"

    os.makedirs(args.path, exist_ok=True)
    print(f"Results will be saved in {args.path}")
    print("DEFAULT_DEVICE " ,DEFAULT_DEVICE)

    #  run dataset test
    run_dataset(
        saliency_model= model,
        rarity_model= rarity_model,
        file_opener= file_opener,
        layers_index= layers_index,
        args= args,
    )
