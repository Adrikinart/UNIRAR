
import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
import torchvision.transforms as transforms

import sys 
import time
sys.path.append('..')
from model import DeepRare
from data import FileOpener
from utils import metrics
import json


if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("cpu")
else:
    DEFAULT_DEVICE = torch.device("cpu")

print("DEFAULT_DEVICE " ,DEFAULT_DEVICE)


def show_images(img, saliency_rare, saliency_rare_details):
    plt.figure(1)

    plt.subplot(421)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Initial Image')

    plt.subplot(422)
    plt.imshow(saliency_rare)
    plt.axis('off')
    plt.title('Final Saliency Map')

    print(saliency_rare_details.shape)

    for i in range(0,saliency_rare_details.shape[0]):

        plt.subplot(4,2,3 + i)
        plt.imshow(saliency_rare_details[i,:, :])
        plt.axis('off')
        plt.title(f'Level {i}Saliency Map')

    plt.show()

def run_dataset(name,directory, model, args, path_save, show = False):


    files = os.listdir(directory)
    opener = FileOpener()
    
    results = []
    start_time_global = time.time()
    for index, filename in enumerate(files):
        go_path = os.path.join(directory, filename)

        # open images
        img = cv2.imread(go_path)
        targ = cv2.imread(go_path.replace("images","targ_labels"),0)
        dist = cv2.imread(go_path.replace("images","dist_labels"),0)

        # open image tensor
        tensor_image = opener.open_image(
            file = go_path , 
            size = (412,412)
        )

        start_time = time.time()
        tensor_image = tensor_image.unsqueeze(0).to(DEFAULT_DEVICE)
        saliency_rare, saliency_rare_details = model(tensor_image)
        end_time = time.time()
        process_time = end_time - start_time

        saliency_rare = saliency_rare.squeeze(0).detach().cpu()
        saliency_rare_details = saliency_rare_details.squeeze(0).squeeze(0).detach().cpu()

        saliency_rare = (saliency_rare - saliency_rare.min()) / (saliency_rare.max() - saliency_rare.min()) * 255

        saliency_rare = cv2.resize(saliency_rare.permute(1, 2, 0).numpy(), (img.shape[1], img.shape[0]))
        saliency_rare = cv2.GaussianBlur(saliency_rare, (5, 5), 0)

        saliency_rare = (saliency_rare).astype(np.uint8)

        if show:
            show_images(
                img,
                saliency_rare,
                saliency_rare_details,
            )

        results.append({
            "filename": filename,
            'path' : directory,
            'metrics' : 
            {
                'saliency_rare': metrics.compute_msr(saliency_rare, targ, dist),
            }
        })

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


    # Save results as JSON file
    print()
    results_path = os.path.join(path_save, f"{name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DeepRare")

    parser.add_argument(
        "--threshold", 
        type=float, 
        default=None, 
        help="Threshold for torch rare 2021"
    )

    args = parser.parse_args()
    model = DeepRare(
        threshold=args.threshold,
    ).to(DEFAULT_DEVICE) # instantiate class

    if args.threshold is None:
        res_dir = os.path.join("..", "res", f"deeprare_noThreshold")
    else:
        res_dir = os.path.join("..", "res", f"deeprare_{args.threshold}")


    os.makedirs(res_dir, exist_ok=True)
    print(f"Results will be saved in {res_dir}")

    run_dataset(
        name= "O3_data" ,
        directory = "/Users/coconut/Documents/Dataset/Saliency/O3_data/images/" , 
        model= model,
        args= args,
        path_save= res_dir
    )
    run_dataset(
        name= "P3_data_sizes" ,
        directory = "/Users/coconut/Documents/Dataset/Saliency/P3_data/sizes/images/" , 
        model= model,
        args= args,
        path_save= res_dir
    )
    run_dataset(
        name= "P3_data_orientations" ,
        directory = "/Users/coconut/Documents/Dataset/Saliency/P3_data/orientations/images/" , 
        model= model,
        args= args,
        path_save= res_dir
    )
    run_dataset(
        name= "P3_data_colors" ,
        directory = "/Users/coconut/Documents/Dataset/Saliency/P3_data/colors/images/" , 
        model= model,
        args= args,
        path_save= res_dir
    )