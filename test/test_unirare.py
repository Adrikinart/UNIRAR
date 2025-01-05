
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
from model import FeatureExtractor, DeepRare, UNIRARE
from data import FileOpener



def post_process(map, img):
    smap = np.exp(map)
    smap = np.squeeze(smap)
    smap = smap
    map_ = (smap / np.amax(smap) * 255).astype(np.uint8)
    return cv2.resize(map_ , (img.shape[1] , img.shape[0]))


def image_inference(model, img : torch.Tensor ) -> np.ndarray :
    img_ = img.unsqueeze(0).unsqueeze(0)
    print(img.dtype)

    map_ = model(img_, source="SALICON")
    return map_.squeeze(0).squeeze(0).squeeze(0).detach().cpu().numpy()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parameters test unisal  video/image")
    parser.add_argument(
        "--type", 
        type=str, 
        default="image", 
        help="image or video"
    )


    parser.add_argument(
        "--source", 
        type=str, 
        default="SALICON", 
        help="SALICON / DFH1K / UCF"
    )


    parser.add_argument(
        "--load", 
        type=str, 
        default="/model/weights/", 
        help="path model to load"
    )


    args = parser.parse_args()

    # create model unirare
    model = UNIRARE(bypass_rnn=False)

    if args.load is not None:
        path_ = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(args.load + "weights_best.pth"):
            model.load_weights( args.load + "weights_best.pth")

    # get images or videos
    directory = ""
    if args.type == "image":
        directory = r'inputs/images/'
        # files = os.listdir(r'inputs/images/')
    elif args.type == 'video':
        directory = r'inputs/videos/'

    files = os.listdir(directory)
    # parcours chaque fichiers
    opener = FileOpener()

    for filename in files:
        print(filename)
        go_path = os.path.join(directory, filename)

        # load FileOpener
        data = None
        if args.type == "image":
            tensor_image = opener.open_image(
                file = go_path , 
                size = (412,412)
            )
            # process image
            img = cv2.imread(go_path)

            # process image
            tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)
            map_, SAL, groups = model(tensor_image, source="SALICON")

            map_ = map_.squeeze(0).squeeze(0).squeeze(0).detach().cpu().numpy()
            map_ = post_process(map_,img)

            SAL = SAL.squeeze(0)
            groups = groups.squeeze(0)

            print(SAL.shape)
            print(groups.shape)


            plt.figure(1)

            plt.subplot(421)
            plt.imshow(img)
            plt.axis('off')
            plt.title('Initial Image')

            plt.subplot(422)
            plt.imshow(SAL)
            plt.axis('off')
            plt.title('Final Saliency Map')

            for i in range(0,groups.shape[-1]):

                plt.subplot(423 + i)
                plt.imshow(groups[:, :, i])
                plt.axis('off')
                plt.title(f'Level {i}Saliency Map')

            plt.show()

            
        elif args.type == 'video':
            data = opener.open_video(
                file = go_path,
                fps = 15,
                seq_len=12,
                size = (412,412)
            )
            # process video frame
        
        
        # if data is None:
        #     continue

        # plt.figure(1)

        # img = cv2.imread(go_path)

        # plt.subplot(421)
        # plt.imshow(img)
        # plt.axis('off')
        # plt.title('Initial Image')

        # plt.subplot(422)
        # plt.imshow(SAL)
        # plt.axis('off')
        # plt.title('Final Saliency Map')

        # for i in range(0,groups.shape[-1]):

        #     plt.subplot(423 + i)
        #     plt.imshow(groups[:, :, i])
        #     plt.axis('off')
        #     plt.title(f'Level {i}Saliency Map')

        # plt.show()