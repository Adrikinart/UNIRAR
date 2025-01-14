
import os
import cv2
import numpy as np
import argparse
import sys 
import time
import json

sys.path.append('..')
from utils import metrics

import torch
from torchvision import transforms
import matplotlib.pyplot as plt

from model import Unisal
from model import RarityNetwork
from model import TranSalNetDense
from model import TranSalNetRes

from model import unisal_file_opener
from model import transal_file_opener

from PIL import Image

if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")


print("DEFAULT_DEVICE " ,DEFAULT_DEVICE)
DIRECTORY = "/Users/coconut/Documents/Dataset/Saliency/SALICON/test/"
# DIRECTORY = "/Users/coconut/Documents/Dataset/Saliency/MIT1003/images/"
# DIRECTORY = "/Users/coconut/Documents/Dataset/Saliency/P3_data/"
# DIRECTORY = "/Users/coconut/Documents/Dataset/Saliency/O3_data/"


def run_model(args, model,file_opener,img_dir):
    start_time = time.time()

    # open image tensor
    tensor_image = file_opener.open_image(
        file = img_dir, 
    )

    saliency, layers = None, None
    if args.model == "Unisal":
        tensor_image = tensor_image.to(DEFAULT_DEVICE)
        saliency,layers = model(tensor_image, source="SALICON",get_all_layers= True)

        saliency = torch.exp(saliency)
        saliency = saliency / torch.amax(saliency)
        saliency= saliency.squeeze(0).squeeze(0)
        layers= layers[0]

    if args.model == "TranSalNetDense" or args.model == "TranSalNetRes":
        tensor_image = tensor_image.to(DEFAULT_DEVICE)
        saliency, layers = model(tensor_image)

        saliency = saliency.squeeze(0)
        toPIL = transforms.ToPILImage()
        saliency = toPIL(saliency)
        saliency = file_opener.postprocess_img(saliency, img_dir)

        saliency= torch.from_numpy(saliency).type(torch.FloatTensor).unsqueeze(0).to(DEFAULT_DEVICE)

    # print(f"Processing {args.model}: {time.time() - start_time:.2f} seconds")

    return saliency, layers

def load_model(args):
    path_ = os.path.dirname(os.path.abspath(__file__))

    if args.model == "Unisal":
        model = Unisal(
            bypass_rnn=False,
        )
        if args.finetune.lower() == "true":

            if os.path.exists(path_ + "/../model/Unisal/weights/" + "weights_best.pth"):
                print("Model Load")
                model.load_weights(path_ + "/../model/Unisal/weights/" + "weights_best.pth")
            else:
                print("Model not found")

    elif args.model == "TranSalNetDense":
        model = TranSalNetDense()
        model.load_state_dict(torch.load(path_ + '/../model/TranSalNet/pretrained_models/TranSalNet_Dense.pth', map_location=DEFAULT_DEVICE))

    elif args.model == "TranSalNetRes":
        model = TranSalNetRes()
        model.load_state_dict(torch.load(path_ + '/../model/TranSalNet/pretrained_models/TranSalNet_Res.pth', map_location=DEFAULT_DEVICE))

    return model

def load_dataloader(args):
    path_ = os.path.dirname(os.path.abspath(__file__))
    print(path_)

    if args.model == "Unisal":
        file_opener = unisal_file_opener
    elif args.model == "TranSalNetDense":
        file_opener = transal_file_opener
    elif args.model == "TranSalNetRes":
        file_opener = transal_file_opener
    return file_opener

def apply_color(map_,img):
    if map_.ndim == 3:
        map_ = map_.squeeze(0)
    map_ = map_.detach().cpu().numpy()

    map_ = cv2.normalize(map_, None, 0, 255, cv2.NORM_MINMAX)
    map_ = cv2.resize(map_, (img.shape[1], img.shape[0]))
    map_ = map_.astype(np.uint8)

    map_color = cv2.applyColorMap(map_, cv2.COLORMAP_JET)
    map_color = cv2.cvtColor(map_color, cv2.COLOR_BGR2RGB)
    map_img = cv2.addWeighted(img, 0.6, map_color, 0.4, 0)

    return map_, map_img


def show_saliency(img, maps= {}, details= None):


    titles= ['Image']
    saliency_colors = [img]
    saliency_colors_images = [img]

    for key, v in maps.items():
        saliency_color, saliency_color_img = apply_color(v, img)
        titles.append(key)
        saliency_colors.append(saliency_color)
        saliency_colors_images.append(saliency_color_img)

    titles_details= []
    colors_details= []
    colors_details_images= []

    if details is not None:
        details= details.squeeze(0)
        for i in range(details.shape[0]):
            detail_color, detail_color_img = apply_color(details[i, :, :], img)
            titles_details.append(f"Detail {i}")
            colors_details.append(detail_color)
            colors_details_images.append(detail_color_img)

    plt.figure(figsize=(10,10))

    rows =  2
    cols = max(len(saliency_colors) , len(colors_details))

    if len(titles_details) != 0:
        rows += 2

    for i in range(cols):
        plt.subplot(rows,cols,i + 1)
        plt.imshow(saliency_colors[i])
        plt.axis('off')
        plt.title(titles[i])


        plt.subplot(rows,cols,cols + i + 1)
        plt.imshow(saliency_colors_images[i])
        plt.axis('off')
        plt.title(titles[i])

    # show details
    for i in range(len(titles_details)):
        plt.subplot(rows,cols,2*cols + i + 1)
        plt.imshow(colors_details[i])
        plt.axis('off')
        plt.title(titles_details[i])

        plt.subplot(rows,cols,3*cols + i + 1)
        plt.imshow(colors_details_images[i])
        plt.axis('off')
        plt.title(titles_details[i])
    

    # output_dir = "./outputs/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # output_path = os.path.join(output_dir, f"image_{index}_{args.name}_finetune_{args.finetune.lower()}_threshold_{args.threshold}.jpeg")
    # plt.savefig(output_path)

    # plt.show()

def run_dataset(args, saliency_model, rarity_model, file_opener):
    files = os.listdir(DIRECTORY)
    start_time_global = time.time()
    
    for index, filename in enumerate(files):
        go_path = os.path.join(DIRECTORY , filename)

        # open images
        img = cv2.imread(go_path)
        if img is None:
            continue

        # run model
        start_time = time.time()
        saliency, layers = run_model(args, saliency_model,file_opener ,go_path)


        if args.model == "Unisal":
            layers_index=[
                [4,5],
                [7,8],
                [10,11],
                [13,14],
                [16,17]
            ]
        elif args.model== "TranSalNetDense":
            layers_index=[
                [2,3],
                [4,5],
                [6,7],
                [9],
                [10]
            ]
        elif args.model == "TranSalNetRes":
            layers_index=[
                [2,3],
                [4,5],
                [6],
                [7],
                [8]
            ]

        # run rarity network
        rarity_map, groups = rarity_model(
            layers_input= layers,
            layers_index=layers_index
        ) 

        process_time = time.time() - start_time


        maps = {
            'saliency': saliency,
            'rarity': rarity_map,
        }


        # create fusion maps with rarity and saliency
        maps['saliency_Add']= rarity_model.add_rarity(saliency, rarity_map)
        maps['saliency_Sub']= rarity_model.sub_rarity(saliency, rarity_map)
        maps['saliency_Prod']= rarity_model.prod_rarity(saliency, rarity_map)
        maps['saliency_Itti']= rarity_model.fuse_rarity(saliency, rarity_map)


        for key, v in maps.items():
            os.makedirs(args.path + "/" +key, exist_ok=True)

            v = v.squeeze(0).squeeze(0).detach().cpu().numpy()
            v = cv2.normalize(v, None, 0, 255, cv2.NORM_MINMAX)
            v = cv2.resize(v, (img.shape[1], img.shape[0]))
            v = v.astype(np.uint8)

            # save images in folder 
            file_name = filename.replace(".jpg" , ".png")
            cv2.imwrite(f"{args.path}/{key}/{file_name}", v)
        # Resize fix and sal to 240x240
        # size = (100,100)
        # fix = cv2.resize(fix, size)
        # sal = cv2.resize(sal, size)

        # Normalize fix and sal between 0 and 1

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

        # print(results[-1])

        # # show results
        # show_saliency(
        #     img= img,
        #     maps= maps,
        #     details= groups
        # )
        # plt.show()



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
        choices= ["Unisal", "TranSalNetDense" , "TranSalNetRes"]
    )

    parser.add_argument(
        "--path", 
        type=str, 
        default="../res/salicon_test",
        help="path to save results"
    )


    # get arguments
    args = parser.parse_args()

    # rarity network
    rarity_model= RarityNetwork(
        threshold=args.threshold
    )

    # load model
    model = load_model(args)

    # load file opener
    file_opener = load_dataloader(args)

    # load model on default device
    rarity_model= rarity_model.to(DEFAULT_DEVICE)
    model = model.to(DEFAULT_DEVICE)
    args.path+= f"_{args.model}_finetune_{args.finetune.lower()}_threshold_{args.threshold}/"

    os.makedirs(args.path, exist_ok=True)
    print(f"Results will be saved in {args.path}")

    #  run dataset test
    run_dataset(
        saliency_model= model,
        rarity_model= rarity_model,
        file_opener= file_opener,
        args= args,
    )
