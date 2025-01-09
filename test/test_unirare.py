
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
from model import UNIRARE
from data import FileOpener
import time



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
        default="../model/weights/", 
        help="path model to load"
    )


    args = parser.parse_args()

    # create model unirare
    model = UNIRARE(
        bypass_rnn=False,
        # layers=args.layers_to_extract
    )

    if args.load is not None:
        path_ = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(args.load + "weights_best.pth"):
            print("Loading model")
            model.load_weights( args.load + "weights_best.pth")
        else:
            print("Model not found")

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
            start_time = time.time()
            saliency, saliency_rare, saliency_rare_details = model(tensor_image, source="SALICON")
            end_time = time.time()
            print(f"Processing time: {end_time - start_time} seconds")

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

            saliency_fusion_sr = np.abs(saliency - saliency_rare)
            saliency_fusion_sr = (saliency_fusion_sr - saliency_fusion_sr.min()) / (saliency_fusion_sr.max() - saliency_fusion_sr.min()) * 255


            saliency = (saliency).astype(np.uint8)
            saliency_rare = (saliency_rare).astype(np.uint8)
            saliency_fusion_add = (saliency_fusion_add).astype(np.uint8)
            saliency_fusion_rs = (saliency_fusion_rs).astype(np.uint8)
            saliency_fusion_sr = (saliency_fusion_sr).astype(np.uint8)




            saliency_fusion_add_color = cv2.applyColorMap(saliency_fusion_add, cv2.COLORMAP_JET)
            saliency_fusion_add_color = cv2.cvtColor(saliency_fusion_add_color, cv2.COLOR_BGR2RGB)

            saliency_fusion_rs_color = cv2.applyColorMap(saliency_fusion_rs, cv2.COLORMAP_JET)
            saliency_fusion_rs_color = cv2.cvtColor(saliency_fusion_rs_color, cv2.COLOR_BGR2RGB)

            saliency_fusion_sr_color = cv2.applyColorMap(saliency_fusion_sr, cv2.COLORMAP_JET)
            saliency_fusion_sr_color = cv2.cvtColor(saliency_fusion_sr_color, cv2.COLOR_BGR2RGB)


            saliency_color = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)
            saliency_color = cv2.cvtColor(saliency_color, cv2.COLOR_BGR2RGB)

            saliency_rare_color = cv2.applyColorMap(saliency_rare, cv2.COLORMAP_JET)
            saliency_rare_color = cv2.cvtColor(saliency_rare_color, cv2.COLOR_BGR2RGB)


            saliency_color_img = cv2.addWeighted(img, 0.6, saliency_color, 0.4, 0)
            saliency_rare_color_img = cv2.addWeighted(img, 0.6, saliency_rare_color, 0.4, 0)

            saliency_fusion_add_color_img = cv2.addWeighted(img, 0.6, saliency_fusion_add_color, 0.4, 0)
            saliency_fusion_rs_color_img = cv2.addWeighted(img, 0.6, saliency_fusion_rs_color, 0.4, 0)
            saliency_fusion_sr_color_img = cv2.addWeighted(img, 0.6, saliency_fusion_sr_color, 0.4, 0)



            print(saliency_rare_details.shape)
            plt.figure(1, figsize=(10,10))

            plt.subplot(451)
            plt.imshow(img)
            plt.axis('off')
            plt.title('Initial Image')

            plt.subplot(452)
            plt.imshow(saliency_rare_color)
            plt.axis('off')
            plt.title('saliency_rare')

            plt.subplot(453)
            plt.imshow(saliency_color)
            plt.axis('off')
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
                
                print(i)
                plt.subplot(4,5,11 + i)
                plt.imshow(saliency_rare_details[i,:, :])
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