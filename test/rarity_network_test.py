
import os
import cv2
import numpy as np
import argparse
import sys 
import time
sys.path.append('..')

import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from model import Unisal
from model import RarityNetwork
from model import TranSalNetDense
from model import TranSalNetRes


from model import unisal_file_opener
from model import transal_file_opener

from PIL import Image
import pysaliency


if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
else:
    DEFAULT_DEVICE = torch.device("cpu")

print("DEFAULT_DEVICE " ,DEFAULT_DEVICE)


def run_model(args, model,file_opener,img_dir):
    start_time = time.time()

    saliency, layers = None, None
    if args.model == "Unisal":
        # open image tensor
        tensor_image = file_opener.open_image(
            file = img_dir, 
        )

        tensor_image = tensor_image.to(DEFAULT_DEVICE)
        saliency,layers = model(tensor_image, source="SALICON",get_all_layers= True)

        saliency = torch.exp(saliency)
        saliency = saliency / torch.amax(saliency)
        saliency= saliency.squeeze(0).squeeze(0)
        layers= layers[0]

    if args.model == "TranSalNetDense" or args.model == "TranSalNetRes":
        # open image tensor
        tensor_image, pad_ = file_opener.open_image(
            file = img_dir, 
        )

        tensor_image = tensor_image.to(DEFAULT_DEVICE)
        saliency, layers = model(tensor_image)

        saliency = saliency.squeeze(0)
        toPIL = transforms.ToPILImage()
        saliency = toPIL(saliency)
        saliency = file_opener.postprocess_img(saliency, img_dir)

        saliency= torch.from_numpy(saliency).unsqueeze(0).to(DEFAULT_DEVICE)
        print(saliency.shape)
        print("ALL LAYERS")
        print(pad_)
        
        new_layers= []
        for layer in layers:
            rw= layer.shape[-1] / pad_['w']
            rh= layer.shape[-2] / pad_['h']

            lpad = int(pad_['left'] * rw) +1
            rpad = int(pad_['right'] * rw)+1
            tpad = int(pad_['top'] * rh)+1
            bpad = int(pad_['bottom'] * rh)+1

            layer = layer[:, :, tpad:layer.shape[-2] - bpad, lpad:layer.shape[-1] - rpad]
            new_layers.append(layer)
        layers= new_layers

    print(f"Processing {args.model}: {time.time() - start_time:.2f} seconds")

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


def show_saliency(img, saliency, rarity_map= None, fusion_add= None, fusion_sub= None, fusion_prod= None, fusion_fuse= None, details= None):

    titles= ['Image']
    saliency_colors = [img]
    saliency_colors_images = [img]

    saliency_color, saliency_color_img = apply_color(saliency, img)
    titles.append("Saliency")
    saliency_colors.append(saliency_color)
    saliency_colors_images.append(saliency_color_img)

    if rarity_map is not None:
        rarity_color, rarity_color_img = apply_color(rarity_map, img)
        titles.append("Rarity")
        saliency_colors.append(rarity_color)
        saliency_colors_images.append(rarity_color_img)

    if fusion_add is not None:
        fusion_add_color, fusion_add_color_img = apply_color(fusion_add, img)
        titles.append("Fusion Add")
        saliency_colors.append(fusion_add_color)
        saliency_colors_images.append(fusion_add_color_img)


    if fusion_sub is not None:
        fusion_sub_color, fusion_sub_color_img = apply_color(fusion_sub, img)
        titles.append("Fusion Sub")
        saliency_colors.append(fusion_sub_color)
        saliency_colors_images.append(fusion_sub_color_img)

    if fusion_prod is not None:
        fusion_prod_color, fusion_prod_color_img = apply_color(fusion_prod, img)
        titles.append("Fusion Prod")
        saliency_colors.append(fusion_prod_color)
        saliency_colors_images.append(fusion_prod_color_img)

    if fusion_fuse is not None:
        fusion_fuse_color, fusion_fuse_color_img = apply_color(fusion_fuse, img)
        titles.append("Fusion Fuse")
        saliency_colors.append(fusion_fuse_color)
        saliency_colors_images.append(fusion_fuse_color_img)

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
    
    plt.show()

    # output_dir = "./outputs/"
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    # output_path = os.path.join(output_dir, f"image_{index}_{args.name}_finetune_{args.finetune.lower()}_threshold_{args.threshold}.jpeg")
    # plt.savefig(output_path)

    # plt.show()


def run_dataset(directory,args, saliency_model, rarity_model, file_opener):
    files = os.listdir(directory)
    
    for index, filename in enumerate(files):
        go_path = os.path.join(directory, filename)

        # open images
        img = cv2.imread(go_path)



        # run model
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
                # [10]
            ]
        elif args.model == "TranSalNetRes":
            layers_index=[
                [2,3],
                [4,5],
                [6],
                [7],
                # [8]
            ]


        # run rarity network
        rarity_map, groups = rarity_model(
            layers_input= layers,
            layers_index=layers_index
        ) 

        # create fusion maps with rarity and saliency
        saliency_add= rarity_model.add_rarity(saliency, rarity_map)
        saliency_sub= rarity_model.sub_rarity(saliency, rarity_map)
        saliency_prod= rarity_model.prod_rarity(saliency, rarity_map)
        saliency_fuse= rarity_model.fuse_rarity(saliency, rarity_map)

        # show results
        show_saliency(
            img= img,
            saliency= saliency,
            rarity_map= rarity_map,
            fusion_add= saliency_add,
            fusion_sub= saliency_sub,
            fusion_prod= saliency_prod,
            fusion_fuse= saliency_fuse,
            details= groups
        )


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

    #  run dataset test
    run_dataset(
        directory = "inputs/images" , 
        saliency_model= model,
        rarity_model= rarity_model,
        file_opener= file_opener,
        args= args,
    )
