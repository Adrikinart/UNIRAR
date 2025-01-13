
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


def show_images(img, saliency_rare, saliency_rare_details, args , index):
    plt.figure(figsize=(10,10))

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


    output_dir = "./outputs/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"image_{index}_{args.name}_threshold_{args.threshold}.jpeg")
    plt.savefig(output_path)
    plt.show()


def run_dataset(name,directory, model, args):
    files = os.listdir(directory)
    opener = FileOpener()
    
    start_time_global = time.time()
    for index, filename in enumerate(files):
        go_path = os.path.join(directory, filename)

        # open images
        img = cv2.imread(go_path)

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

        show_images(
            img,
            saliency_rare,
            saliency_rare_details,
            args,
            index
        )

        # Print loading bar with FPS information
        process_time_global = time.time() - start_time_global

        total = len(files)
        progress = (index + 1) / total
        bar_length = 40
        block = int(round(bar_length * progress))
        text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}% | {index}/{total} | Time: {process_time:.4f}s"
        sys.stdout.write(text)
        sys.stdout.flush()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DeepRare")

    parser.add_argument(
        "--threshold", 
        type=float, 
        default=None, 
        help="Threshold for torch rare 2021"
    )


    parser.add_argument(
        "--name", 
        type=str, 
        default="deeprare", 
        help="add name information"
    )

    args = parser.parse_args()
    model = DeepRare(
        threshold=args.threshold,
    ).to(DEFAULT_DEVICE) # instantiate class

    run_dataset(
        name= "test innputs" ,
        directory = "./inputs/images/" , 
        model= model,
        args= args
    )