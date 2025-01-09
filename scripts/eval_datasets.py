
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
DEFAULT_DEVICE = torch.device("cpu")

print("DEFAULT_DEVICE " ,DEFAULT_DEVICE)

def load_model(args, weights_path):
    if args.model == "unirare":
        model = UNIRARE(bypass_rnn=False)
    elif args.model == "unirare_finetuned":
        model = UNIRARE(bypass_rnn=False)
        if os.path.exists(weights_path + "weights_best.pth"):
            print("Loading model")
            model.load_weights( weights_path + "weights_best.pth")
        else:
            print("Model not found")
    elif args.model == "unisal":
        model = UNISAL(bypass_rnn=False)

        if os.path.exists(weights_path + "weights_best.pth"):
            model.load_weights(weights_path + "weights_best.pth")
    elif args.model == "deeprare":
        print("DEEP RARE")
        model = DeepRare(
            threshold=args.threshold,
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

def run_model(model, image, model_name, image_raw):
    img = image_raw
    tensor_image = image.to(DEFAULT_DEVICE)
    if model_name == "unirare" or model_name == "unirare_finetuned":
        tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)
        saliency, saliency_rare, saliency_rare_details = model(tensor_image, source="SALICON")


        saliency= saliency.squeeze(0).squeeze(0)
        saliency_rare = saliency_rare.squeeze(0).detach().cpu()
        saliency_rare_details = saliency_rare_details.squeeze(0).squeeze(0).detach().cpu()

        saliency = post_process(saliency.detach().cpu().numpy(),img)
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min()) * 255
        saliency_rare = (saliency_rare - saliency_rare.min()) / (saliency_rare.max() - saliency_rare.min()) * 255

        saliency = cv2.resize(saliency, (img.shape[1], img.shape[0]))
        saliency_rare = cv2.resize(saliency_rare.permute(1, 2, 0).numpy(), (img.shape[1], img.shape[0]))
        saliency_rare = cv2.GaussianBlur(saliency_rare, (5, 5), 0)

        saliency_fusion_add = saliency_rare + saliency
        saliency_fusion_add = (saliency_fusion_add - saliency_fusion_add.min()) / (saliency_fusion_add.max() - saliency_fusion_add.min()) * 255

        saliency_fusion_rs = np.abs(saliency_rare - saliency)
        saliency_fusion_rs = (saliency_fusion_rs - saliency_fusion_rs.min()) / (saliency_fusion_rs.max() - saliency_fusion_rs.min()) * 255

        saliency = (saliency).astype(np.uint8)
        saliency_rare = (saliency_rare).astype(np.uint8)
        saliency_fusion_add = (saliency_fusion_add).astype(np.uint8)
        saliency_fusion_rs = (saliency_fusion_rs).astype(np.uint8)

        # Normalize sal between 0 and 255
        saliency_rare = cv2.normalize(saliency_rare, None, 0, 255, cv2.NORM_MINMAX)

        return {
            'saliency' : saliency,
            'saliency_rare': saliency_rare,
            'saliency_rare_details' : saliency_rare_details,
            'saliency_fusion_add' : saliency_fusion_add,
            'saliency_fusion_rs' : saliency_fusion_rs
        }


    elif model_name == "deeprare":
        tensor_image = tensor_image.unsqueeze(0)
        saliency_rare, saliency_rare_details  = model(tensor_image)

        saliency_rare = saliency_rare.squeeze(0).detach().cpu()
        saliency_rare_details = saliency_rare_details.squeeze(0).squeeze(0).detach().cpu()

        saliency_rare = (saliency_rare - saliency_rare.min()) / (saliency_rare.max() - saliency_rare.min()) * 255
        saliency_rare = cv2.resize(saliency_rare.permute(1, 2, 0).numpy(), (img.shape[1], img.shape[0]))
        saliency_rare = cv2.GaussianBlur(saliency_rare, (5, 5), 0)

        return {
            'saliency' : None,
            'saliency_rare': saliency_rare,
            'saliency_rare_details' : saliency_rare_details,
            'saliency_fusion_add' : None,
            'saliency_fusion_rs' : None
        }

    
    elif model_name == "unisal":
        tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)

        saliency_map = model(tensor_image, source="SALICON")
        saliency_map = saliency_map.squeeze(0).squeeze(0).detach().cpu().numpy()
        saliency_map = post_process(saliency_map, tensor_image)

        return {
            'saliency' : saliency_map,
            'saliency_rare': None,
            'saliency_rare_details' : None,
            'saliency_fusion_add' : None,
            'saliency_fusion_rs' : None
        }
    
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
    

def run_dataset(dataset_name,dataset, model, model_name, path_folder):
    results = []
    total = len(dataset)-1
    start_time = time.time()

    # Create a folder with name based on args information
    result_folder_path = os.path.join(path_folder, dataset_name)

    # # Create the folder if it doesn't exist
    # if not os.path.exists(result_folder_path):
    #     os.makedirs(result_folder_path)


    for i in range(total):
        img, targ, dist, _, file_name = dataset[i]

        # run model 
        start_time_model = time.time()
        saliency_images = run_model(model, img, model_name,cv2.imread(file_name))

        # plt.figure(figsize=(12,12))
        # plt.subplot(351)
        # plt.imshow(cv2.imread(file_name))
        # plt.axis('off')
        # plt.title('Image')

        # plt.subplot(356)
        # plt.imshow(cv2.imread(file_name))
        # plt.axis('off')
        # plt.title('Image')
        
        # compute metrics
        index = 1

        result = {}

        for k , sal in saliency_images.items():
            if sal is None:
                continue

            if k == 'saliency_rare_details':
                continue

            # # subplot
            # plt.subplot(3,5,1+ index)
            # plt.imshow(sal)
            # plt.axis('off')
            # plt.title(k)

            # # subplot
            # plt.subplot(3,5,6+ index)
            # plt.imshow(sal)
            # plt.axis('off')
            # plt.title(k + 'img')

            # resize sal
            sal = cv2.resize(sal, (targ.shape[-2], targ.shape[-1]))

            # Normalize sal between 0 and 255
            sal = cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX)

            # # compute metrics 
            msr = metrics.compute_msr(sal, targ.squeeze(0).numpy(), dist.squeeze(0).numpy())

            # # Add time process for each image
            msr['process_time'] = time.time() - start_time_model
            msr['file_name'] = f"image_result_{i}.png"
            result[k] = msr

        results.append(result)

        process_time = time.time() - start_time
        # Print loading bar with FPS information
        progress = (i + 1) / total
        bar_length = 40
        block = int(round(bar_length * progress))
        fps = (i + 1) / process_time
        text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}% | FPS: {fps:.2f} | {i}/{total} | Time: {process_time:.2f}s"
        sys.stdout.write(text)
        sys.stdout.flush()
        
        # plt.savefig(os.path.join(result_folder_path, f"image_result_{i}.jpg"))
        # plt.show()
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
        default="deeprare", 
        choices=["unirare", "unirare_finetuned", "unisal", "deeprare"],
        help="Select model to use"
    )

    parser.add_argument(
        "--P3Dataset", 
        type=str, 
        default="/Users/coconut/Documents/Dataset/Saliency/P3_data/", 
        # default="C:/Users/lelon/Documents/Dataset/P3_data/", 

        help="path model to load"
    )

    parser.add_argument(
        "--O3Dataset", 
        type=str, 
        default="/Users/coconut/Documents/Dataset/Saliency/O3_data/", 
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
        # default="4,5,8, 9,11,12,13,  16,17,18,19,  26,27,28,29", 
        default="4 , 5, 11 , 14", 
        help="Liste d'entiers séparés par des virgules (ex: 1,2,3)."
    )

    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.7, 
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

    # Create a folder with name based on args information
    result_folder = f"results_{args.model}_{args.type}"
    result_folder_path = os.path.join("../res", result_folder)

    # Create the folder if it doesn't exist
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    # load dataloaders
    o3_dataset = O3Dataset(
        path =args.O3Dataset,
        input_size=(args.input_size,args.input_size),
    )

    p3_dataset_sizes = P3Dataset(
        path =args.P3Dataset + "sizes/",
        input_size=(args.input_size,args.input_size),
    )

    p3_dataset_orientations = P3Dataset(
        path =args.P3Dataset + "orientations/",
        input_size=(args.input_size,args.input_size),
    )

    p3_dataset_colors = P3Dataset(
        path =args.P3Dataset + "colors/",
        input_size=(args.input_size,args.input_size),
    )

    # Run dataset and collect results
    results['O3Dataset'] = run_dataset("O3Dataset",o3_dataset, model, args.model,path_folder= result_folder_path)
    results['P3Dataset_sizes'] = run_dataset('P3Dataset_sizes',p3_dataset_sizes, model, args.model,path_folder= result_folder_path)
    results['P3Dataset_orientations'] = run_dataset('P3Dataset_orientations',p3_dataset_orientations, model, args.model,path_folder= result_folder_path)
    results['P3Dataset_colors'] = run_dataset('P3Dataset_colors',p3_dataset_colors, model, args.model,path_folder= result_folder_path)

    # Save results to JSON file
    json_file_path = os.path.join(result_folder_path, "results.json")
    with open(json_file_path, "w") as json_file:
        json.dump(results, json_file, indent=4)
    print("Results saved to:", json_file_path)