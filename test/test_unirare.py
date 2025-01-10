
import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch
import torchvision.models as models
import torchvision.transforms as transforms

import sys 
import time
sys.path.append('..')
from model import UNIRARE
from data import FileOpener
from utils import metrics
import json

if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")

print("DEFAULT_DEVICE " ,DEFAULT_DEVICE)

def post_process(map, img):
    smap = np.exp(map)
    smap = np.squeeze(smap)
    smap = smap
    map_ = (smap / np.amax(smap) * 255).astype(np.uint8)
    return cv2.resize(map_ , (img.shape[1] , img.shape[0]))


def show_images(img, saliency, saliency_rare, saliency_fusion_add, saliency_fusion_rs, saliency_rare_details,args ,index):

    saliency_fusion_add_color = cv2.applyColorMap(saliency_fusion_add, cv2.COLORMAP_JET)
    saliency_fusion_add_color = cv2.cvtColor(saliency_fusion_add_color, cv2.COLOR_BGR2RGB)

    saliency_fusion_rs_color = cv2.applyColorMap(saliency_fusion_rs, cv2.COLORMAP_JET)
    saliency_fusion_rs_color = cv2.cvtColor(saliency_fusion_rs_color, cv2.COLOR_BGR2RGB)

    saliency_color = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
    saliency_color = cv2.cvtColor(saliency_color, cv2.COLOR_BGR2RGB)

    saliency_rare_color = cv2.applyColorMap(saliency_rare, cv2.COLORMAP_JET)
    saliency_rare_color = cv2.cvtColor(saliency_rare_color, cv2.COLOR_BGR2RGB)


    saliency_color_img = cv2.addWeighted(img, 0.6, saliency_color, 0.4, 0)
    saliency_rare_color_img = cv2.addWeighted(img, 0.6, saliency_rare_color, 0.4, 0)

    saliency_fusion_add_color_img = cv2.addWeighted(img, 0.6, saliency_fusion_add_color, 0.4, 0)
    saliency_fusion_rs_color_img = cv2.addWeighted(img, 0.6, saliency_fusion_rs_color, 0.4, 0)

    plt.figure(figsize=(10,10))

    plt.subplot(451)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Initial Image')

    plt.subplot(452)
    plt.imshow(saliency_rare)
    plt.axis('off')
    plt.title('saliency_rare')

    plt.subplot(453)
    plt.axis('off')
    plt.imshow(saliency_color)
    plt.title('saliency')

    plt.subplot(454)
    plt.imshow(saliency_fusion_add_color)
    plt.axis('off')
    plt.title('fusion_add')

    plt.subplot(455)
    plt.imshow(saliency_fusion_rs_color)
    plt.axis('off')
    plt.title('fusion_rs')

    plt.subplot(456)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Initial Image')

    plt.subplot(457)
    plt.imshow(saliency_rare_color_img)
    plt.axis('off')
    plt.title('saliency_rare')

    plt.subplot(458)
    plt.imshow(saliency_color_img)
    plt.axis('off')
    plt.title('saliency')

    plt.subplot(459)
    plt.imshow(saliency_fusion_add_color_img)
    plt.axis('off')
    plt.title('fusion_add')

    plt.subplot(4,5,10)
    plt.imshow(saliency_fusion_rs_color_img)
    plt.axis('off')
    plt.title('fusion_rs')

    for i in range(0,saliency_rare_details.shape[0]):
        plt.subplot(4,5,11 + i)
        plt.imshow(saliency_rare_details[i,:, :])
        plt.axis('off')
        plt.title(f'Level {i}Saliency Map')


    output_dir = "./outputs/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, f"image_{index}_deeprare_finetune_{args.finetune.lower()}_threshold_{args.threshold}.jpeg")
    plt.savefig(output_path)

    plt.show()


def run_dataset(name,directory, model, args):


    files = os.listdir(directory)
    opener = FileOpener()
    
    start_time_global = time.time()
    for index, filename in enumerate(files):
        go_path = os.path.join(directory, filename)

        # load FileOpener
        if args.type == "image":

            # open images
            img = cv2.imread(go_path)

            # open image tensor
            tensor_image = opener.open_image(
                file = go_path , 
                size = (412,412)
            )

            # process image
            tensor_image = tensor_image.unsqueeze(0).unsqueeze(0).to(DEFAULT_DEVICE)
            start_time = time.time()
            saliency, saliency_rare, saliency_rare_details = model(tensor_image, source="SALICON")
            end_time = time.time()
            process_time = end_time - start_time
            # print(f"Processing time: {process_time:.2f} seconds")

            saliency= saliency.squeeze(0).squeeze(0).detach().cpu()
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

            show_images(
                img,
                saliency,
                saliency_rare,
                saliency_fusion_add,
                saliency_fusion_rs,
                saliency_rare_details,
                args,
                index
            )


        total = len(files)
        progress = (index + 1) / total
        bar_length = 40
        block = int(round(bar_length * progress))
        text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}% | {index}/{total} | Time: {process_time:.4f}s"
        sys.stdout.write(text)
        sys.stdout.flush()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters test unisal  video/image")
    parser.add_argument(
        "--type", 
        type=str, 
        default="image", 
        help="image or video"
    )

    parser.add_argument(
        "--threshold", 
        type=float, 
        default=None, 
        help="Threshold for torch rare 2021"
    )

    parser.add_argument(
        "--source", 
        type=str, 
        default="SALICON", 
        help="SALICON / DFH1K / UCF"
    )

    parser.add_argument(
        "--finetune", 
        type=str, 
        default="true",
        help="true for finetune, false for no finetune",
        choices = ["true", "false"]
    )

    args = parser.parse_args()

    # create model unirare
    model = UNIRARE(
        bypass_rnn=False,
        threshold=args.threshold
    ).to(DEFAULT_DEVICE)

    if args.finetune.lower() == "true":
        path_ = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists("../model/weights/" + "weights_best.pth"):
            print("Loading model")
            model.load_weights( "../model/weights/" + "weights_best.pth")
        else:
            print("Model not found")

    run_dataset(
        name= "test" ,
        directory = "inputs/images" , 
        model= model,
        args= args,
    )
